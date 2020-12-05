import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from greedy_base_hGLM import Greedy_Base_hGLM
from sklearn import metrics
from tqdm import tqdm,tnrange



class Greedy_Search:
    def __init__(self, V_ref, train_T, test_T, T_no, E_neural, I_neural,
                batch_size, batch_no, max_sub, cell_type):

        self.train_T = train_T
        self.test_T = test_T
        self.T_no = T_no
        self.batch_size = batch_size
        self.max_sub = max_sub
        self.cell_type = cell_type
        self.batch_no = batch_no

        self.E_no = E_neural.shape[1]
        self.I_no = I_neural.shape[1]

        self.train_V_ref = V_ref[:train_T].float()
        self.test_V_ref = V_ref[train_T:train_T+test_T].float().cuda()

        self.train_S_E = E_neural[:train_T].float()
        self.train_S_I = I_neural[:train_T].float()
        self.test_S_E = E_neural[train_T:train_T+test_T].float().cuda()
        self.test_S_I = I_neural[train_T:train_T+test_T].float().cuda()

        self.E_no = E_neural.shape[1]
        self.I_no = I_neural.shape[1]

    def make_C_den(self, raw):
        sub_no = raw.shape[0] + 1
        C_den = torch.zeros(sub_no, sub_no)
        for i in range(sub_no - 1):
            C_den[raw[i], i+1] = 1
        return C_den

    def make_batch_idx(self, epoch_no = 1):
        batch_no = (self.train_V_ref.shape[0] - self.batch_size) * epoch_no
        train_idx = np.empty((epoch_no, self.train_V_ref.shape[0] - self.batch_size))
        for i in range(epoch_no):
            part_idx = np.arange(self.train_V_ref.shape[0] - self.batch_size)
            np.random.shuffle(part_idx)
            train_idx[i] = part_idx
        train_idx = train_idx.flatten()
        train_idx = torch.from_numpy(train_idx)
        return train_idx

    def train(self, C_den):
        sub_no = C_den.shape[0]
        change_idx = torch.where(C_den[:,-1] == 1)[0]

        train_idx = self.make_batch_idx(epoch_no=1)

        model = Greedy_Base_hGLM(C_den.cuda(), self.E_no, self.I_no, self.T_no)
        model = model.float().cuda()
        
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)
        
        for i in tnrange(self.batch_no):
            model.train()
            optimizer.zero_grad()

            batch_idx = train_idx[i].long()
            batch_S_E = self.train_S_E[batch_idx : batch_idx+self.batch_size].float().cuda()
            batch_S_I = self.train_S_I[batch_idx : batch_idx+self.batch_size].float().cuda()
            batch_pred = model(batch_S_E, batch_S_I)
            batch_ref = self.train_V_ref[batch_idx:batch_idx+self.batch_size].cuda()
                
            diff = (batch_ref - batch_pred) ** 1
            loss = torch.var(diff)

            loss.backward()
            optimizer.step()
            scheduler.step()

            avg_var_exp = 0
            count = 0
            
            if (i%50 == 49) & (i >= self.batch_no - 20*50):
                count += 1
                test_pred = model(self.test_S_E, self.test_S_I)
                test_score = metrics.explained_variance_score(y_true=self.test_V_ref.cpu().detach().numpy(),
                                                      y_pred=test_pred.cpu().detach().numpy(),
                                                      multioutput='uniform_average')
                avg_var_exp += test_score

            if i == self.batch_no-1:
                torch.save(model.state_dict(), "/media/hdd01/sklee/greedy/greedybaseGLM_"+self.cell_type+"_sub"+str(sub_no)+"-"+str(change_idx.item())+".pt")

        avg_var_exp /= count
        return avg_var_exp

    def greedy_search(self):
        best_score = torch.zeros(self.max_sub-1)

        for i in tnrange(self.max_sub - 1):
            if i == 0:
                C_den_raw = torch.tensor([0])
                C_den = self.make_C_den(C_den_raw)
                avg_var_exp = self.train(C_den)
                best_score[i] = avg_var_exp

            else:
                var_exp_list = torch.zeros(i+1)
                for j in range(i+1):
                    new_C_den_raw = torch.zeros(i+1)
                    new_C_den_raw[:-1] = C_den_raw
                    new_C_den_raw[-1] = j
                    new_C_den_raw = new_C_den_raw.long()
                    new_C_den = self.make_C_den(new_C_den_raw)
                    avg_var_exp = self.train(new_C_den)
                    var_exp_list[j] = avg_var_exp
                    print("DONE: Sub"+str(i+2)+"-"+str(j))
                    

                chosen_idx = torch.argmax(var_exp_list)
                chosen_C_den_raw = torch.zeros(i+1)
                chosen_C_den_raw[:-1] = C_den_raw
                chosen_C_den_raw[-1] = chosen_idx
                C_den_raw = chosen_C_den_raw.long()
                best_score[i] = var_exp_list[chosen_idx]
                print("Sub"+str(i+2)+"_scores", var_exp_list)

            print("FINAL C_DEN_RAW", C_den_raw)
            print("FINAL_BEST_SCORES", best_score)

        #return C_den_raw, best_score
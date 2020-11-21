import torch
from torch import nn
from torch.nn import functional as F
import shGLM_parts as parts

class LVAE_shGLM(nn.Module):
    def __init__(self, C_den, C_syn_e, C_syn_i, T_syn, syn_basis_no,
                T_hist, hist_basis_no, hid_dim, fix_var, T_V, theta_spike_init, W_spike_init):
        super().__init__()

        self.C_den = C_den
        self.C_syn_e = C_syn_e
        self.C_syn_i = C_syn_i
        self.T_syn = T_syn
        self.syn_basis_no = syn_basis_no
        self.T_hist = T_hist
        self.hist_basis_no = hist_basis_no
        self.sub_no = C_den.shape[0]
        self.fix_var = fix_var
        self.T_V = T_V
        self.hid_dim = hid_dim
        self.theta_spike_init = theta_spike_init
        self.W_spike_init = W_spike_init

        self.out_idx_list = []
        self.in_idx_list = []
        self.idx_count = 0
        self.layer_no = 0
        while self.idx_count < self.sub_no:
            if self.idx_count == 0:
                self.out_idx = 0
                self.out_idx_list.append(torch.tensor([0]).cuda())
                self.in_idx = torch.sort(torch.where(C_den[self.out_idx,:] == 1)[0])[0]
                self.in_idx_list.append(self.in_idx)
                self.idx_count += 1
                self.layer_no += 1
            else:
                self.out_idx = self.in_idx_list[self.layer_no - 1]
                self.out_idx_list.append(self.out_idx)
                self.in_idx = torch.sort(torch.where(C_den[self.out_idx,:] == 1)[1])[0]
                self.in_idx_list.append(self.in_idx)
                self.idx_count += torch.numel(self.out_idx)
                self.layer_no += 1
        
        self.middle_no = self.layer_no - 2
        if self.middle_no > 0:
            self.middle_decoder_list = []
            self.middle_encoder_list = []
            for i in range(self.middle_no):
                self.part_C_den = self.C_den[self.out_idx_list[i+1], self.in_idx_list[i+1]]
                self.middle_decoder_list.append(parts.Middle_Integ(self.part_C_den, self.syn_basis_no, self.T_hist,
                                                        self.hist_basis_no, self.fix_var, self.theta_spike_init,
                                                                  self.W_spike_init))
                if i == 0:
                    self.middle_encoder_list.append(MLP((self.T_V-1)*2 + 1, self.hid_dim, torch.numel(self.in_idx_list[i])))
                else:
                    self.middle_encoder_list.append(MLP(self.hid_dim, self.hid_dim, torch.numel(self.in_idx_list[i])))
            
            self.middle_decoder_list = nn.ModuleList(self.middle_decoder_list.reverse())
            self.middle_encoder_list = nn.ModuleList(self.middle_encoder_list)
                                           
        self.root_C_den = self.C_den[self.out_idx_list[0], self.in_idx_list[0]].reshape(1,-1)

        self.leaf_decoder = parts.Leaf_Integ(torch.numel(self.out_idx_list[-1]), self.syn_basis_no, self.T_hist, self.hist_basis_no, self.fix_var, self.theta_spike_init, self.W_spike_init)
        self.root_decoder = parts.Root_Integ(self.root_C_den, self.syn_basis_no, self.T_hist)
        
        if self.middle_no > 0:
            self.leaf_encoder = parts.MLP(self.hid_dim, self.hid_dim, torch.numel(self.out_idx_list[-1]))
        else:
            self.leaf_encoder = parts.MLP((self.T_V-1)*2 + 1, self.hid_dim, torch.numel(self.out_idx_list[-1]))

        self.spike_convolve = parts.Spike_Convolve(self.C_syn_e, self.C_syn_i, self.T_syn, self.syn_basis_no)

    def Encoder(self, V_in):
        T_data = V_in.shape[0]
        pad_V_in = torch.zeros(T_data + 2* (self.T_V - 1)).cuda()
        pad_V_in[self.T_V-1 : self.T_V-1+T_data] = pad_V_in[self.T_V-1 : self.T_V-1+T_data]+ V_in
        up_mu_array = torch.zeros(T_data, self.sub_no - 1).cuda()

        for t in range(T_data):
            up_mu_list = torch.zeros(self.sub_no - 1).cuda()
            count = 0

            if self.middle_no > 0:
                for i in range(self.middle_no):
                    if i == 0:
                        mid_enc, up_mu = self.middle_encoder_list[i].encode(pad_V_in[t:t + 2*self.T_V - 1])
                        up_mu_list[count:count+torch.numel(self.in_idx_list[i])] = up_mu_list[count:count+torch.numel(self.middle_encoder_list[i])] + up_mu
                        count += torch.numel(self.in_idx_list[i])
                    else:
                        mid_enc, up_mu = self.middle_encoder_list[i].encode(mid_enc)
                        up_mu_list[count:count+torch.numel(self.in_idx_list[i])] = up_mu_list[count:count+torch.numel(self.middle_encoder_list[i])] + up_mu
                        count += torch.numel(self.in_idx_list[i])
                _, up_mu = self.leaf_encoder.encode(mid_enc)
                up_mu_list[count:torch.numel(self.out_idx_list[-1])] = up_mu_list[count:torch.numel(self.out_idx_list[-1])] + up_mu
                up_mu_array[t] = up_mu_array[t] + up_mu_list
                
            else:
                _, up_mu = self.leaf_encoder.encode(pad_V_in[t:t + 2*self.T_V - 1])
                up_mu_list[-torch.numel(self.out_idx_list[-1]):] = up_mu_list[-torch.numel(self.out_idx_list[-1]):] + up_mu
                up_mu_array[t] = up_mu_array[t] + up_mu_list
        
        return up_mu_array

    def Decoder(self, S_e, S_i, up_mu_array=None):

        T_data = S_e.shape[0]
        S_conv = self.spike_convolve(S_e, S_i)
        posterior_mu_array = torch.zeros(T_data, self.sub_no - 1).cuda() ## Comb of down and up
        down_mu_array = torch.zeros(T_data, self.sub_no-1).cuda() ## Pure down
        
        if up_mu_array == None:
            hid_Y, hid_Z, mu_Z, down_mu_Z = self.leaf_decoder(S_conv[:,-torch.numel(self.out_idx_list[-1]):])
            posterior_mu_array[:,-torch.numel(self.out_idx_list[-1]):] = posterior_mu_array[:,-torch.numel(self.out_idx_list[-1]):] + mu_Z
            down_mu_array[:,-torch.numel(self.out_idx_list[-1]):] = down_mu_array[:,-torch.numel(self.out_idx_list[-1]):] + down_mu_Z

            if self.middle_no > 0:
                for i in range(self.middle_no):
                    hid_Y, hid_Z, mu_Z, down_mu_Z = self.middle_decoder_list[i](S_conv[:, self.out_idx_list[-i-2]],
                                                                hid_Y, hid_Z)
                    posterior_mu_array[:,self.out_idx_list[-i-2]] = posterior_mu_array[:,self.out_idx_list[-i-2]] + mu_Z
                    down_mu_array[:,self.out_idx_list[-i-2]] = down_mu_array[:,self.out_idx_list[-i-2]] + down_mu_Z
            
        else:          
            hid_Y, hid_Z, mu_Z, down_mu_Z = self.leaf_decoder(S_conv[:,-torch.numel(self.out_idx_list[-1]):], up_mu_array[:,-torch.numel(self.out_idx_list[-1]):])
            posterior_mu_array[:,-torch.numel(self.out_idx_list[-1]):] = posterior_mu_array[:,-torch.numel(self.out_idx_list[-1]):] + mu_Z
            down_mu_array[:,-torch.numel(self.out_idx_list[-1]):] = down_mu_array[:,-torch.numel(self.out_idx_list[-1]):] + down_mu_Z

            if self.middle_no > 0:
                for i in range(self.middle_no):
                    hid_Y, hid_Z, mu_Z, down_mu_Z = self.middle_decoder_list[i](S_conv[:, self.out_idx_list[-i-2]],
                                                                hid_Y, hid_Z, up_mu_array[:, self.out_idx_list[-i-2]])
                    posterior_mu_array[:,self.out_idx_list[-i-2]] = posterior_mu_array[:,self.out_idx_list[-i-2]] + mu_Z
                    down_mu_array[:,self.out_idx_list[-i-2]] = down_mu_array[:,self.out_idx_list[-i-2]] + down_mu_Z

        final_Y = self.root_decoder(S_conv[:, 0], hid_Y, hid_Z)
        
        #print(hid_Y[300:315])
        #print(final_Y[300:315])

        return final_Y, posterior_mu_array, down_mu_array

    def loss(self, V_in, S_e, S_i, beta):
        up_mu_array = self.Encoder(V_in)
        final_V, posterior_mu_array, down_mu_array = self.Decoder(S_e, S_i, up_mu_array)

        rec_loss = torch.mean((V_in - final_V) ** 2)
        
        #diff = (final_V - V_in) ** 1
        #rec_loss = torch.var(diff)
        
        kl_element_wise = 0.5 * ((posterior_mu_array - down_mu_array).pow(2) / self.fix_var)
        kl_loss = torch.mean(torch.sum(kl_element_wise, 0))
        #loss = rec_loss + beta * kl_loss

        post_prob = torch.mean(torch.sigmoid(posterior_mu_array + torch.randn(V_in.shape[0], self.sub_no-1).cuda()*self.fix_var**(0.5)),0)
        down_prob = torch.mean(torch.sigmoid(down_mu_array + torch.randn(V_in.shape[0], self.sub_no-1).cuda()*self.fix_var**(0.5)) ,0)
        
        post_mu = torch.mean(posterior_mu_array, 0)
        down_mu = torch.mean(down_mu_array, 0)

        return rec_loss, kl_loss, final_V, post_prob, down_prob, post_mu, down_mu
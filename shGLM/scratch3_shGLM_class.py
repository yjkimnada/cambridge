import torch
import torch.nn as nn
import torch.nn.functional as F

class shGLM(nn.Module):
    def __init__(self, C_den, C_syn, syn_T, Ensyn,
                 syn_basis_no, hist_basis_no, spike_status, hist_T):
        super(shGLM, self).__init__()
        
        self.sub_no = C_den.shape[0]
        self.C_den = C_den
        self.C_syn = C_syn
        self.Ensyn = Ensyn
        self.syn_basis_no = syn_basis_no
        self.hist_basis_no = hist_basis_no
        self.spike_status = spike_status
        
        ### Synaptic Parameters ###
        self.V_o = nn.Parameter(torch.ones(1)*(-69), requires_grad=True)
        self.K_syn_raw = torch.ones(self.sub_no, self.syn_basis_no, 2)
        self.K_syn_raw[:,:,0] = 0.04
        self.K_syn_raw[:,:,1] = -0.02
        self.K_syn = nn.Parameter(self.K_syn_raw, requires_grad=True)
        self.syn_basis_tau = nn.Parameter(torch.arange(1.6, 1.6+self.syn_basis_no), requires_grad=True)
        self.Delta = nn.Parameter(torch.ones(1) * 2, requires_grad=True)
        self.C = nn.Parameter(torch.ones(self.sub_no) * 1 , requires_grad=True)
        self.Theta = nn.Parameter(torch.zeros(self.sub_no), requires_grad=True)
        self.syn_T = syn_T
        
        ### Spike History Parameters ###
        self.thresh = nn.Parameter(torch.ones(self.sub_no)*85, requires_grad=True)
        self.thresh_scale = nn.Parameter(torch.ones(self.sub_no)*100, requires_grad=True)
        self.spike_size = nn.Parameter(torch.ones(self.sub_no)*4, requires_grad=True)
        self.hist_basis_tau = nn.Parameter(torch.arange(1.7, 1.6+self.hist_basis_no*0.5,step=0.5), requires_grad=True)
        self.K_hist = nn.Parameter(torch.ones(self.sub_no, self.hist_basis_no) * (0.025), requires_grad=True)
        self.hist_T = hist_T

    def forward(self, X):
        X_T = X.shape[0]
        
        ######
        ## Convolve the 1 or 3 synaptic kernels based on synapse location on dendrite
        ## Also scale with W
        ######
        
        subunit_in = torch.empty(X.shape[0], self.sub_no).cuda()
        # contains just the synaptic inputs weighted by W & kernels (No nonlin, non subunit constant)

        t_e = torch.arange(self.syn_T).cuda()
        t_i = torch.arange(self.syn_T).cuda()
        delta = torch.exp(self.Delta)
        t_e = t_e - delta
        t_i = t_i - delta
        t_e[t_e < 0.0] = 0.0
        t_i[t_i < 0.0] = 0.0
        
        for s in range(self.sub_no):
            e_idx = torch.where(self.C_syn[s] == 1)[0]
            i_idx = torch.where(self.C_syn[s] == -1)[0]
            in_e = X[:, e_idx].T.unsqueeze(1)
            in_i = X[:, i_idx].T.unsqueeze(1)

            full_e_kern = torch.zeros(self.syn_T).cuda()
            full_i_kern = torch.zeros(self.syn_T).cuda()

            for b in range(self.syn_basis_no):
                tau = torch.exp(self.syn_basis_tau[b])
                t_e_tau = t_e / tau
                t_i_tau = t_i /tau
                part_e_kern = t_e_tau * torch.exp(-t_e_tau)
                part_i_kern = t_i_tau * torch.exp(-t_i_tau)
                full_e_kern = full_e_kern + part_e_kern * self.K_syn[s,b,0]
                full_i_kern = full_i_kern + part_i_kern * self.K_syn[s,b,1]

            full_e_kern = full_e_kern.reshape(1,1,-1)
            full_i_kern = full_i_kern.reshape(1,1,-1)
            filtered_e = F.conv1d(in_e, full_e_kern, padding=int(self.syn_T//2))
            filtered_i = F.conv1d(in_i, full_i_kern, padding=int(self.syn_T//2))
            filtered_e = filtered_e.squeeze(1).T
            filtered_i = filtered_i.squeeze(1).T
            subunit_in[:,s] = torch.sum(filtered_e, 1) + torch.sum(filtered_i, 1)

        ######
        ## Group subunits' inputs and process nonlinearities and subunit constants
        ######
        
        subunit_out = torch.empty(X.shape[0], self.sub_no).cuda()
        # each column will contain the subunit outputs (after nonlin AND subunit weight)
            
        root_idx = torch.flip(torch.unique(self.C_den, sorted=True), dims=[0]) - 1
        root_idx = root_idx[:-1] #adjust subunits numbers to indices and remove -1
        root_idx = root_idx.long()
        
        #############################
        if self.spike_status == False:
            for m in range(self.sub_no):
                subunit = self.sub_no - 1 - m
                if subunit not in root_idx:
                    subunit_out[:,subunit] = torch.sigmoid(subunit_in[:,subunit] - self.Theta[subunit]) * torch.exp(self.C[subunit])

                elif subunit in root_idx: # r is actual root subunit index (not number)
                    s_idx = torch.where(self.C_den == subunit+1)[0]
                    subunit_out[:,subunit] = subunit_in[:,subunit]
                    for k in s_idx: # k is actual subunit index
                        subunit_out[:,subunit] = subunit_out[:,subunit] + subunit_out[:,k]
                    subunit_out[:,subunit] = torch.sigmoid(subunit_out[:,subunit] - self.Theta[subunit]) * torch.exp(self.C[subunit])

        ##############################
        elif self.spike_status == True:
            for m in range(self.sub_no):
                subunit = self.sub_no - 1 - m
                spike_hist = torch.zeros(self.hist_T).cuda()
                
                if subunit not in root_idx:
                    sub_in = subunit_in[:,subunit]
                    final_sub_out = torch.empty(sub_in.shape[0]).cuda()
                    
                    t_h = torch.arange(self.hist_T).cuda()
                    full_kern = torch.zeros(self.hist_T).cuda()
                    for b in range(self.hist_basis_no):
                        tau = torch.exp(self.hist_basis_tau[b])
                        t_tau = t_h / tau
                        part_kern = t_tau * torch.exp(-t_tau)
                        full_kern = full_kern + part_kern*self.K_hist[subunit,b]
                    
                    for t in range(X_T):
                        sub_thresh_pre = sub_in[t] + torch.matmul(full_kern, spike_hist)
                        sub_thresh_post = torch.sigmoid(sub_thresh_pre - self.Theta[subunit])
                        sub_spike = torch.sigmoid(sub_thresh_post*self.thresh_scale[subunit] - self.thresh[subunit])
                        final_sub_out[t] = sub_thresh_post + sub_spike * self.spike_size[subunit]
                        spike_hist = torch.cat((sub_spike.reshape(1,1), spike_hist[:-1].reshape(-1,1)),0).flatten()
                        #print(spike_hist)
                    subunit_out[:,subunit] = final_sub_out * torch.exp(self.C[subunit]) 
                    
                elif subunit in root_idx:
                    s_idx = torch.where(self.C_den == subunit+1)[0]
                    sub_in = subunit_in[:,subunit]
                    for k in s_idx:
                        sub_in = sub_in + subunit_out[:,k]
                    final_sub_out = torch.empty(sub_in.shape[0]).cuda()
                    
                    t_h = torch.arange(self.hist_T).cuda()
                    full_kern = torch.zeros(self.hist_T).cuda()
                    for b in range(self.hist_basis_no):
                        tau = torch.exp(self.hist_basis_tau[b])
                        t_tau = t_h / tau
                        part_kern = t_tau * torch.exp(-t_tau)
                        full_kern = full_kern + part_kern*self.K_hist[subunit,b]

                    for t in range(X_T):
                        sub_thresh_pre = sub_in[t] + torch.matmul(full_kern, spike_hist)
                        sub_thresh_post = torch.sigmoid(sub_thresh_pre - self.Theta[subunit])
                        sub_spike = torch.sigmoid(sub_thresh_post*self.thresh_scale[subunit] - self.thresh[subunit])
                        final_sub_out[t] = sub_thresh_post + sub_spike * self.spike_size[subunit]
                        spike_hist = torch.cat((sub_spike.reshape(1,1), spike_hist[:-1].reshape(-1,1)),0).flatten()
                    subunit_out[:,subunit] = final_sub_out * torch.exp(self.C[subunit])
              
        final_out = subunit_out[:,0] + self.V_o
        
        return final_out
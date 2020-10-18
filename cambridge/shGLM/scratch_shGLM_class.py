import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class shGLM(nn.Module):
    def __init__(self, C_den, C_syn, W_size, G_syn, E_no,
                 hist_basis_tau, spike_status, hist_T):
        super(shGLM, self).__init__()
        # C_den is shape (M)
        # C_syn is shape (M, N). 0, 1, -1 for none, excit, inhib
        # W is window size for synaptic kernel
        
        self.sub_no = C_den.shape[0]
        self.W_size = W_size
        self.C_den = C_den
        self.C_syn = C_syn
        self.Ensyn = Ensyn
        self.G_syn = G_syn
        self.G_no = int(torch.max(G_syn).item()+1)
        self.spike_status = spike_status
        self.hist_basis_no = hist_basis_tau.shape[0]
        
        self.V_o = nn.Parameter(torch.ones(1)*(-69), requires_grad=True)
        self.Delta = nn.Parameter(torch.ones(1) * 0.5, requires_grad=True)
        self.W_raw = torch.ones(self.G_no)
        self.W_raw[:self.G_no//2] = 2
        self.W_raw[self.G_no//2:] = -1
        self.W = nn.Parameter(self.W_raw * 0.05, requires_grad=True)
        self.C = nn.Parameter(torch.ones(self.sub_no) * 1 , requires_grad=True)
        self.Theta = nn.Parameter(torch.zeros(self.sub_no), requires_grad=True)
        
        self.thresh = nn.Parameter(torch.ones(self.sub_no)*60, requires_grad=True)
        self.thresh_scale = nn.Parameter(torch.ones(self.sub_no)*100, requires_grad=False)
        self.spike_size = nn.Parameter(torch.ones(self.sub_no)*4, requires_grad=True)
        self.hist_T = hist_T
        self.K_hist = nn.Parameter(torch.ones(self.sub_no, self.hist_basis_no)*(-0.05), requires_grad=True)
        
    def forward(self, X):
        X_T = X.shape[0]
        
        ######
        ## Convolve the 1 or 3 synaptic kernels based on synapse location on dendrite
        ## Also scale with W
        ######
        filtered_in = torch.empty(X.shape[0], X.shape[1]).cuda()
        
        for g in range(self.G_no):
            g_idx = torch.where(self.G_syn == g)[0]
            in_g = X[:,g_idx].T.unsqueeze(1)
            
            t_g = torch.arange(self.W_size).cuda()
            delta_g = torch.exp(self.Delta)
            t_g = t_g - delta_g
            t_g[t_g < 0.0] = 0.0
            
            t_tau_g_fast = t_g / torch.exp(self.Tau[g])
            t_tau_g_slow = t_g / (torch.exp(self.Tau[g])*2.8+10.4)
            kern_g_fast = t_tau_g_fast * torch.exp(-t_tau_g_fast)
            kern_g_slow = t_tau_g_slow * torch.exp(-t_tau_g_slow) * 0.3
            kern_g = (kern_g_slow + kern_g_fast)/1.3
            
            kern_g = kern_g.reshape(1,1,-1)
            filtered = F.conv1d(in_g, kern_g, padding=int(self.W_size//2))
            filtered = filtered.squeeze(1).T
            filtered_in[:,g_idx] = filtered * self.W[g]
            
        ######
        ## Group subunits' inputs and process nonlinearities and subunit constants
        ######
        
        subunit_in = torch.empty(X.shape[0], self.sub_no).cuda()
        # contains just the synaptic inputs weighted by W & kernels (No nonlin, non subunit constant)
        subunit_out = torch.empty(X.shape[0], self.sub_no).cuda()
        # each column will contain the subunit outputs (after nonlin AND subunit weight)
        
        for s in range(self.sub_no):
            e_idx = torch.where(self.C_syn[s] > 0)[0]
            i_idx = torch.where(self.C_syn[s] < 0)[0]
            in_e = filtered_in[:,e_idx]
            in_i = filtered_in[:,i_idx]
            sum_in_e = torch.sum(in_e, 1)
            sum_in_i = torch.sum(in_i, 1)
            subunit_in[:,s] = sum_in_e + sum_in_i
            
        root_idx = torch.flip(torch.unique(self.C_den, sorted=True), dims=[0]) - 1
        root_idx = root_idx[:-1] #adjust subunits numbers to indices and remove -1
        root_idx = root_idx.long()
        
        #############################
        if spike_status == False:
            for s in range(self.sub_no):
                if s not in root_idx:
                    subunit_out[:,s] = torch.sigmoid(subunit_in[:,s] - self.Theta[s]) * torch.exp(self.C[s])

                elif r in root_idx: # r is actual root subunit index (not number)
                    s_idx = torch.where(self.C_den == s+1)[0]
                    subunit_out[:,s] = subunit_in[:,s]
                    for k in r_idx: #k is actual subunit index
                        subunit_out[:,s] = subunit_out[:,s] + subunit_out[:,k]
                    subunit_out[:,s] = torch.sigmoid(subunit_out[:,s] - self.Theta[s]) * torch.exp(self.C[s])
            
        ##############################
        elif spike_status == True:
            for s in range(self.sub_no):
                subunit = self.sub_no - 1 - s
                spike_hist = torch.zeros(self.hist_T).cuda()
                
                if s not in root_idx:
                    sub_out = torch.sigmoid(subunit_in[:,s] - self.Theta[s])
                    t_h = torch.arange(self.hist_T).cuda()
                    full_kern = torch.zeros(self.hist_T).cuda()
                    for b in range(self.hist_basis_no):
                        tau = self.hist_basis_tau[b]
                        t_tau = t_h / tau
                        part_kern = t_tau * torch.exp(-t_tau)
                        full_kern = full_kern + part_kern*self.K_hist[s,b]
                    
                    for t in range(X_T):
                        sub_thresh = sub_out[t] + full_kern * spike_hist
                        sub_out[t] = sub_thresh
                        sub_spike = torch.sigmoid(sub_thresh)
                        sub_out[t] = sub_out[t] + sub_spike * self.spike_size[s]
                        spike_hist = torch.cat(sub_spike, spike_hist[:-1])
                    subunit_out[:,s] = sub_out * torch.exp(self.C[s])
                    
                elif s in root_idx:
                    s_idx = torch.where(self.C_den == s+1)[0]
                    subunit_out[:,s] = subunit_in[:,s]
                    for k in s_idx:
                        subunit_out[:,s] = subunit_out[:,s] + subunit_out[:,k]
                    
                    sub_out = torch.sigmoid(subunit_in[:,s] - self.Theta[s]) 
                    t_h = torch.arange(self.hist_T).cuda()
                    full_kern = torch.zeros(self.hist_T).cuda()
                    for b in range(self.hist_basis_no):
                        tau = self.hist_basis_tau[b]
                        t_tau = t_h / tau
                        part_kern = t_tau * torch.exp(-t_tau)
                        full_kern = full_kern + part_kern*self.K_hist[s,b]
                    
                    for t in range(X_T):
                        sub_thresh = sub_out[t] + full_kern * spike_hist
                        sub_out[t] = sub_thresh
                        sub_spike = torch.sigmoid(sub_thresh)
                        sub_out[t] = sub_out[t] + sub_spike * self.spike_size[s]
                        spike_hist = torch.cat(sub_spike, spike_hist[:-1])
                    subunit_out[:,s] = sub_out * torch.exp(self.C[s])
              
            final_out = subunit_out[:,0] + self.V_o     
        
        return final_out
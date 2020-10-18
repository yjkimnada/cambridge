import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class hGLM(nn.Module):
    def __init__(self, C_den, C_syn, B, G, Ensyn):
        super(hGLM, self).__init__()
        # C_den is shape (M)
        # C_syn is shape (M, N). 0, 1, -1 for none, excit, inhib
        # B is window size for synaptic kernel
        
        self.M = C_den.shape[0]
        self.B = B
        self.C_den = C_den
        self.C_syn = C_syn
        self.Ensyn = Ensyn
        self.G = G
        self.G_no = int(torch.max(G).item()+1)
        
        self.V_o = nn.Parameter(torch.ones(1)*(-69), requires_grad=True)
        
        self.Tau_raw = torch.ones(self.G_no)
        self.Tau_raw[:self.G_no//2] = 1
        self.Tau_raw[self.G_no//2:] = 1
        self.Tau = nn.Parameter(self.Tau_raw * 1.4, requires_grad=True)
        #self.Tau_ratio = nn.Parameter(torch.ones(1) * 6.5, requires_grad = True)
        #self.K_ratio = nn.Parameter(torch.ones(1) * 0.5, requires_grad=True)
        self.Delta = nn.Parameter(torch.ones(1) * 0.25, requires_grad=True)
        
        self.W_raw = torch.ones(self.G_no)
        self.W_raw[:self.G_no//2] = 2
        self.W_raw[self.G_no//2:] = -1
        self.W = nn.Parameter(self.W_raw * 0.05, requires_grad=True)
        self.C = nn.Parameter(torch.ones(self.M) * 1 , requires_grad=True)
        self.Theta = nn.Parameter(torch.zeros(self.M), requires_grad=True)
        
    def forward(self, X):
        # X is size (T, N=E+I)
        
        ######
        ## Convolve the 1 or 3 synaptic kernels based on synapse location on dendrite
        ## Also scale with W
        ######
        filtered_in = torch.empty(X.shape[0], X.shape[1]).cuda()
        
        for g in range(self.G_no):
            g_idx = torch.where(self.G == g)[0]
            in_g = X[:,g_idx].T.unsqueeze(1)
            
            t_g = torch.arange(self.B).cuda()
            delta_g = torch.exp(self.Delta)
            t_g = t_g - delta_g
            t_g[t_g < 0.0] = 0.0
            
            
            if g < self.G_no//2:
                t_tau_g_fast = t_g / torch.exp(self.Tau[g])
                t_tau_g_slow = t_g / (torch.exp(self.Tau[g])*2.8+10.4)
                
                kern_g_fast = t_tau_g_fast * torch.exp(-t_tau_g_fast)
                kern_g_slow = t_tau_g_slow * torch.exp(-t_tau_g_slow) * 0.3

                kern_g = (kern_g_slow + kern_g_fast)/1.3
            else:
                t_tau_g = t_g / torch.exp(self.Tau[g])
                kern_g = t_tau_g * torch.exp(-t_tau_g)
            """
            t_tau_g_fast = t_g / torch.exp(self.Tau[g])
            t_tau_g_slow = t_g / (torch.exp(self.Tau[g])* self.Tau_ratio)
                
            kern_g_fast = t_tau_g_fast * torch.exp(-t_tau_g_fast)
            kern_g_slow = t_tau_g_slow * torch.exp(-t_tau_g_slow) * 0.3

            kern_g = (kern_g_slow + kern_g_fast)/1.3
            """
            
            kern_g = kern_g.reshape(1,1,-1)
            
            filtered = F.conv1d(in_g, kern_g, padding=int(self.B//2))
            filtered = filtered.squeeze(1).T
            filtered_in[:, g_idx] = filtered * self.W[g]
            
        ######
        ## Group subunits' inputs and process nonlinearities and subunit constants
        ######
        subunit_in = torch.empty(X.shape[0], self.M).cuda()
        # contains just the synaptic inputs weighted by W (No nonlin, non subunit constant)
        subunit_out = torch.empty(X.shape[0], self.M).cuda()
        # each column will contain the subunit outputs (after nonlin AND subunit weight)
        
        for m in range(self.M):
            e_idx = torch.where(self.C_syn[m] > 0)[0]
            i_idx = torch.where(self.C_syn[m] < 0)[0]
            in_e = filtered_in[:,e_idx]
            in_i = filtered_in[:,i_idx]
            sum_in_e = torch.sum(in_e, 1)
            sum_in_i = torch.sum(in_i, 1)
            subunit_in[:,m] = sum_in_e + sum_in_i
        
        root_idx = torch.flip(torch.unique(self.C_den, sorted=True), dims=[0]) - 1
        root_idx = root_idx[:-1] #adjust subunits numbers to indices and remove -1
        root_idx = root_idx.long()
        
        for m in range(self.M):
            if m not in root_idx:
                subunit_out[:,m] = torch.sigmoid(subunit_in[:,m] - self.Theta[m]) * torch.exp(self.C[m])
        
        for r in root_idx: # r is actual root subunit index (not number)
            r_idx = torch.where(self.C_den == r+1)[0]
            subunit_out[:,r] = subunit_in[:,r]
            for k in r_idx: #k is actual subunit index
                subunit_out[:,r] = subunit_out[:,r] + subunit_out[:,k]
            subunit_out[:,r] = torch.sigmoid(subunit_out[:,r] - self.Theta[r]) * torch.exp(self.C[r])
                                
        final_out = subunit_out[:,0] + self.V_o

        return final_out
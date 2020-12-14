
  
import torch
from torch import nn
from torch.nn import functional as F

class Greedy_Base_hGLM(nn.Module):
    def __init__(self, C_den, syn_loc_e, syn_loc_i, E_no, I_no, T_no, kern_no):
        super().__init__()

        self.C_den = C_den
        self.T_no = T_no
        self.sub_no = C_den.shape[0]
        self.E_no = E_no
        self.I_no = I_no
        self.kern_no = kern_no
        self.syn_loc_e = syn_loc_e
        self.syn_loc_i = syn_loc_i

        ### Synapse Parameters ###
        self.W_syn = nn.Parameter(torch.randn(self.kern_no, 2) * 0.1, requires_grad=True)
        self.Tau_syn = nn.Parameter(torch.ones(self.kern_no, 2) , requires_grad=True)
        self.Delta_syn = nn.Parameter(torch.rand(self.kern_no, 2), requires_grad=True)
        
        ### Ancestor Subunit Parameters ###
        self.W_sub = nn.Parameter(torch.rand(self.sub_no) , requires_grad=True)

        ### Subunit Output Parameters ###
        self.V_o = nn.Parameter(torch.randn(1), requires_grad=True)
        self.Theta = nn.Parameter(torch.zeros(self.sub_no), requires_grad=True)

        ### C_syn Parameters ###
        self.C_syn_e_logit = nn.Parameter(torch.ones(self.sub_no, self.E_no), requires_grad=True)
        self.C_syn_i_logit = nn.Parameter(torch.ones(self.sub_no, self.I_no), requires_grad=True)

    def forward(self, S_e, S_i, temp, test):
        T_data = S_e.shape[0]

        ### Construct C_syn_e, C_syn_i

        if test == True:
            C_syn_e = torch.zeros_like(self.C_syn_e_logit).cuda()
            C_syn_i = torch.zeros_like(self.C_syn_i_logit).cuda()
            for i in range(C_syn_e.shape[1]):
                idx = torch.argmax(self.C_syn_e_logit[:,i])
                C_syn_e[idx,i] = 1
            for i in range(C_syn_i.shape[1]):
                idx = torch.argmax(self.C_syn_i_logit[:,i])
                C_syn_i[idx,i] = 1
        
        elif test == False:
            u_e = torch.rand_like(self.C_syn_e_logit).cuda()
            u_i = torch.rand_like(self.C_syn_i_logit).cuda()
            eps = 1e-8
            g_e = -torch.log(- torch.log(u_e + eps) + eps)
            g_i = -torch.log(- torch.log(u_i + eps) + eps)
            raw_C_syn_e = F.softmax((self.C_syn_e_logit + g_e) / temp, dim=0)
            raw_C_syn_i = F.softmax((self.C_syn_i_logit + g_i) / temp, dim=0)
            
            C_syn_e = raw_C_syn_e
            C_syn_i = raw_C_syn_i

        ### Make Filters
        kern_e = torch.zeros(self.E_no, self.T_no).cuda()
        kern_i = torch.zeros(self.I_no, self.T_no).cuda()

        for k in range(self.kern_no):
            e_idx = torch.where(self.syn_loc_e == k)[0]
            i_idx = torch.where(self.syn_loc_i == k)[0]
            t_raw = torch.arange(self.T_no).cuda()

            delta_e = torch.exp(self.Delta_syn[k,0])
            delta_i = torch.exp(self.Delta_syn[k,1])

            t_e = t_raw - delta_e
            t_i = t_raw - delta_i
            t_e[t_e < 0.0] = 0.0
            t_i[t_i < 0.0] = 0.0
            
            tau_e = torch.exp(self.Tau_syn[k,0])
            tau_i = torch.exp(self.Tau_syn[k,1])
            t_e_tau = t_e / tau_e
            t_i_tau = t_i / tau_i
            kern_e[e_idx,:] = t_e_tau * torch.exp(-t_e_tau) * self.W_syn[k,0]
            kern_i[i_idx,:] = t_i_tau * torch.exp(-t_i_tau) * self.W_syn[k,1]

        pad_S_e = torch.zeros(T_data + self.T_no - 1, self.E_no).cuda()
        pad_S_i = torch.zeros(T_data + self.T_no - 1, self.I_no).cuda()
        pad_S_e[-T_data:,:] = pad_S_e[-T_data:,:] + S_e
        pad_S_i[-T_data:,:] = pad_S_i[-T_data:,:] + S_i
        
        kern_e = kern_e.unsqueeze(1)
        kern_i = kern_i.unsqueeze(1)
        kern_e = torch.flip(kern_e, [2])
        kern_i = torch.flip(kern_i, [2])
        pad_S_e = pad_S_e.T.unsqueeze(0)
        pad_S_i = pad_S_i.T.unsqueeze(0)
        
        filtered_e = F.conv1d(pad_S_e, kern_e, groups = self.E_no).squeeze(0).T
        filtered_i = F.conv1d(pad_S_i, kern_i, groups = self.I_no).squeeze(0).T

        syn_in_e = torch.matmul(filtered_e, C_syn_e.T)
        syn_in_i = torch.matmul(filtered_i, C_syn_i.T)
        syn_in = syn_in_e + syn_in_i

        ### Solve for Y_t
        sub_out = torch.zeros(T_data, self.sub_no).cuda()
        
        for s in range(self.sub_no):
            sub_idx = -s-1
            leaf_idx = torch.where(self.C_den[sub_idx] == 1)[0]

            if torch.numel(leaf_idx) == 0:
                nonlin_out = torch.tanh(syn_in[:,sub_idx]) # (T_data,) 
                sub_out[:,sub_idx] = sub_out[:,sub_idx] + nonlin_out
            else:
                leaf_in = sub_out[:,leaf_idx] * self.W_sub[leaf_idx] # (T_data,)
                nonlin_in = syn_in[:,sub_idx] + torch.sum(leaf_in, 1) # (T_data,)
                nonlin_out = torch.tanh(nonlin_in)
                sub_out[:,sub_idx] = sub_out[:,sub_idx] + nonlin_out
        
        final_voltage = sub_out[:,0]*self.W_sub[0] + self.V_o

        return final_voltage


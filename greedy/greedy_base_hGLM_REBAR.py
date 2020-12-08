import torch
from torch import nn
from torch.nn import functional as F

def H(x):
    # x is shape (sub_no, syn_no):
    out = torch.zeros_like(x).cuda()
    
    for s in range(x.shape[1]):
        max_idx = torch.argmax(x[:,s])
        out[max_idx,s] = 1
        
    return out
        

def reparam_pz(u, theta):
    z = torch.log(theta) - torch.log(-torch.log(u))
    return z


def reparam_pz_b(v, b, theta):
    #z_squiggle = torch.zeros_like(v).cuda()
    
    same_idx = torch.where(b == 1)
    diff_ones = torch.ones_like(v).cuda()
    same_ones = torch.zeros_like(v).cuda()
    diff_ones[same_idx[0], same_idx[1]] = 0
    same_ones[same_idx[0], same_idx[1]] = 1
    
    z_diff_raw = -torch.log(v)/theta
    z_diff_subtract = torch.log(v[same_idx[0], same_idx[1]]).reshape(1,-1)
    z_diff_raw = z_diff_raw - z_diff_subtract
    z_diff = diff_ones * -torch.log(z_diff_raw)
    
    z_same = same_ones * -torch.log(-torch.log(v))
    
    z_squiggle = z_same + z_diff
    
    
    
    
    for s in range(v.shape[1]):
        k_idx = torch.where(b[:,s] == 1)[0]
        v_k = v[k_idx,s]
        
        for k in range(v.shape[0]):
            if k == k_idx:
                z_squiggle[k,s] = -torch.log(-torch.log(v_k))
            else:
                z_squiggle[k,s] = -torch.log(-torch.log(v[k,s])/theta[k,s] - torch.log(v_k))
                
    return z_squiggle





class Greedy_Base_hGLM(nn.Module):
    def __init__(self, C_den, E_no, I_no, T_no):
        super().__init__()

        self.C_den = C_den
        self.T_no = T_no
        self.sub_no = C_den.shape[0]
        self.E_no = E_no
        self.I_no = I_no

        ### Synapse Parameters ###
        self.W_syn = nn.Parameter(torch.randn(self.sub_no, 2) * 0.1, requires_grad=True)
        self.Tau_syn = nn.Parameter(torch.ones(self.sub_no, 2) , requires_grad=True)
        self.Delta_syn = nn.Parameter(torch.rand(self.sub_no, 2), requires_grad=True)
        
        ### Ancestor Subunit Parameters ###
        self.W_sub = nn.Parameter(torch.rand(self.sub_no) , requires_grad=True)

        ### Subunit Output Parameters ###
        self.V_o = nn.Parameter(torch.randn(1), requires_grad=True)
        self.Theta = nn.Parameter(torch.zeros(self.sub_no), requires_grad=True)

        ### C_syn Parameters ###
        #self.C_syn_e_log = nn.Parameter(torch.ones(self.sub_no, self.E_no), requires_grad=True)
        #self.C_syn_i_log = nn.Parameter(torch.ones(self.sub_no, self.I_no), requires_grad=True)
        self.C_syn_log = nn.Parameter(torch.ones(self.sub_no, self.E_no + self.I_no), requires_grad=True)

    def evaluate_f(self, C_syn, S_e, S_i):
        T_data = S_e.shape[0]
        
        C_syn_e = C_syn[:,:self.E_no].clone()
        C_syn_i = C_syn[:,-self.I_no:].clone()
        
        ### Pre-convolve synapse inputs
        syn_in = torch.zeros(T_data, self.sub_no).cuda()
        
        for s in range(self.sub_no):
            t_raw = torch.arange(self.T_no).cuda()

            delta_e = torch.exp(self.Delta_syn[s,0])
            delta_i = torch.exp(self.Delta_syn[s,1])

            t_e = t_raw - delta_e
            t_i = t_raw - delta_i
            t_e[t_e < 0.0] = 0.0
            t_i[t_i < 0.0] = 0.0
            
            tau_e = torch.exp(self.Tau_syn[s,0])
            tau_i = torch.exp(self.Tau_syn[s,1])
            t_e_tau = t_e / tau_e
            t_i_tau = t_i / tau_i
            full_e_kern = t_e_tau * torch.exp(-t_e_tau) * self.W_syn[s,0]
            full_i_kern = t_i_tau * torch.exp(-t_i_tau) * self.W_syn[s,1]

            in_e = torch.matmul(S_e, C_syn_e.T[:,s])
            in_i = torch.matmul(S_i, C_syn_i.T[:,s])
            pad_in_e = torch.zeros(T_data + self.T_no - 1).cuda()
            pad_in_i = torch.zeros(T_data + self.T_no - 1).cuda()
            pad_in_e[-T_data:] = pad_in_e[-T_data:] + in_e
            pad_in_i[-T_data:] = pad_in_i[-T_data:] + in_i
            pad_in_e = pad_in_e.reshape(1,1,-1)
            pad_in_i = pad_in_i.reshape(1,1,-1)

            full_e_kern = torch.flip(full_e_kern, [0])
            full_i_kern = torch.flip(full_i_kern, [0])
            full_e_kern = full_e_kern.reshape(1,1,-1)
            full_i_kern = full_i_kern.reshape(1,1,-1)
            
            filtered_e = F.conv1d(pad_in_e, full_e_kern, padding=0).squeeze(1).T
            filtered_i = F.conv1d(pad_in_i, full_i_kern, padding=0).squeeze(1).T

            syn_in[:,s] = syn_in[:,s] + filtered_e.flatten() + filtered_i.flatten()

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
        
    
    def forward(self, S_e, S_i):
        
        
        ### Sample random uniform U, V ###
        u = torch.rand(self.sub_no, self.E_no+self.I_no).cuda()
        v = torch.rand(self.sub_no, self.E_no+self.I_no).cuda()
        
        ### Construct C_syn_e, C_syn_i
        C_syn_theta = F.softmax(self.C_syn_log, 0)
        
        rebar_z = reparam_pz(u, C_syn_theta)
        hard_z = H(rebar_z) ## Discrete C_syn
        rebar_zb = reparam_pz_b(v, hard_z, C_syn_theta)
        soft_z = torch.sigmoid(rebar_z/0.5) + 1e-9 ## Soft C_syn
        soft_zb = torch.sigmoid(rebar_zb/0.5)+ 1e-9
        
        ### Evaluate f ###
        V_hard_z = self.evaluate_f(hard_z, S_e, S_i)
        V_soft_z = self.evaluate_f(soft_z, S_e, S_i)
        V_soft_zb = self.evaluate_f(soft_zb, S_e, S_i)

        return V_hard_z, V_soft_z, V_soft_zb, C_syn_theta, hard_z, soft_z, soft_zb
        
        
        
        
    

        


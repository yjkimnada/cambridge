import torch
import torch.nn as nn
import torch.nn.functional as F

class GP_hGLM(nn.Module):
    def __init__(self, C_den, sub_no, N, M, R):
        super().__init__()

        self.C_den = C_den
        self.sub_no = sub_no
        self.S = 2*sub_no # 2x subunits 
        self.N = N
        self.M = M
        self.R = R

        # DSE Kernel Parameters
        self.alpha_log = nn.Parameter(torch.randn(self.S)*0.01 - 12, requires_grad=True) # POSITIVE
        self.beta = nn.Parameter(torch.zeros(self.S) , requires_grad=True)
        self.gamma_log = nn.Parameter(torch.randn(self.S)*0.01 - 12, requires_grad=True) # POSITIVE
        self.kern_var_log = nn.Parameter(torch.randn(self.S)*0.01 + 5, requires_grad=True) # POSITIVE
        #self.delta = nn.Parameter(torch.randn(self.S) , requires_grad=True)
        #self.eta = nn.Parameter(torch.randn(self.S) , requires_grad=True)

        # U Prior Parameters
        self.mean_u = nn.Parameter(torch.zeros(self.S, self.M), requires_grad=True)
        self.S_u_lower_vals = nn.Parameter(torch.randn(self.S, self.M*(self.M+1)//2)*0.1+1.1, requires_grad=True)

        # Input Vectors
        self.f_in = torch.arange(self.N).cuda()
        self.u_in = nn.Parameter(torch.arange(0,self.N//self.M*self.M ,self.N//self.M).double().reshape(1,-1).repeat(self.S,1), requires_grad=True)
        
        # Between Subunit Parameters
        self.W_log = nn.Parameter(torch.randn(self.sub_no) , requires_grad=True) # POSITIVE

        ### Subunit Output Parameters ###
        self.V_o = nn.Parameter(torch.randn(1), requires_grad=True)
        self.Theta = nn.Parameter(torch.zeros(self.sub_no), requires_grad=True)

    def DSE_kernel(self, row_in, col_in, s_idx):
        alpha = torch.exp(self.alpha_log[s_idx])
        gamma = torch.exp(self.gamma_log[s_idx])
        kern_var = torch.exp(self.kern_var_log[s_idx])
        beta = self.beta[s_idx]

        term1 = alpha * (row_in - beta)**2
        term2 = gamma * (row_in - col_in)**2
        term3 = alpha * (col_in - beta)**2

        out = kern_var * torch.exp(- term1 - term2 - term3)
        return out
    
    def forward(self, S_e, S_i):
        T = S_e.shape[0]

        #----- GP for Filters -----#

        # Prepare Q_U prior covariance #
        all_S_u = torch.zeros(self.S, self.M, self.M).cuda()
        M_lower_idx = torch.tril_indices(self.M, self.M).cuda()
        for s in range(self.S):
            S_u_lower = torch.zeros(self.M, self.M).double().cuda()
            S_u_lower[M_lower_idx[0], M_lower_idx[1]] = S_u_lower[M_lower_idx[0], M_lower_idx[1]] + self.S_u_lower_vals[s]
            S_u = torch.matmul(S_u_lower, S_u_lower.T)
            all_S_u[s] = all_S_u[s] + S_u

        # Prepare P_U covariance #
        all_cov_u = torch.zeros(self.S, self.M, self.M).cuda()
        for s in range(self.S):
            cov_u = self.DSE_kernel(self.u_in[s].reshape(-1,1) , self.u_in[s].reshape(1,-1), s)
            all_cov_u[s] = all_cov_u[s] + cov_u
        
        all_cov_u_inv = torch.inverse(all_cov_u)

        # Prepare F covariance #
        all_cov_f = torch.zeros(self.S, self.N, self.N).cuda()
        for s in range(self.S):
            cov_f = self.DSE_kernel(self.f_in[s].reshape(-1,1) , self.f_in[s].reshape(1,-1), s)
            all_cov_f[s] = all_cov_f[s] + cov_f
        
        # Prepare F-U covariance #
        # (N, M)
        all_cov_f_u = torch.zeros(self.S, self.N, self.M).cuda()
        for s in range(self.S):
            cov_f_u = self.DSE_kernel(self.f_in[s].reshape(-1,1) , self.u_in[s].reshape(1,-1), s)
            all_cov_f_u[s] = all_cov_f_u[s] + cov_f_u

        # Prepare joint FU prior mean, covariance #
        all_mu = torch.zeros(self.S, self.N).cuda()
        #all_sigma = torch.zeros(self.S, self.N, self.N).cuda()

        for s in range(self.S):
            K_n_m = all_cov_f_u[s]
            K_m_m_inv = all_cov_u_inv[s]
            #K_n_n = all_cov_f[s]
            #S_u = all_S_u[s]
            m_u = self.mean_u[s]

            A = torch.matmul(K_n_m , K_m_m_inv).double()
            mu = torch.matmul(A , m_u)
            all_mu[s] = all_mu[s] + mu

            #A_K_m_n = torch.matmul(A, K_n_m.T)
            #A_S = torch.matmul(A, S_u)
            #A_S_AT = torch.matmul(A_S , A.T)
            #sigma = K_n_n - A_K_m_n + A_S_AT
            #all_sigma[s] = all_sigma[s] + sigma
                  
        # Sample f for each subunit R times #
        # JUST THE POSTERIOR MEAN?!

        all_F = all_mu

        #----- Spike Convolve -----#

        F_e = all_F[:self.sub_no].unsqueeze(1)
        F_i = all_F[self.sub_no:].unsqueeze(1)
        flip_F_e = torch.flip(F_e, [2])
        flip_F_i = torch.flip(F_i, [2])
        #flip_F_e = F_e
        #flip_F_i = F_i

        pad_S_e = torch.zeros(T + self.N-1, self.sub_no).cuda()
        pad_S_i = torch.zeros(T + self.N-1, self.sub_no).cuda()
        pad_S_e[-T:] = pad_S_e[-T:] + S_e
        pad_S_i[-T:] = pad_S_i[-T:] + S_i
        pad_S_e = pad_S_e.T.unsqueeze(0)
        pad_S_i = pad_S_i.T.unsqueeze(0)

        filtered_e = F.conv1d(pad_S_e, flip_F_e, padding=0, groups=self.sub_no).squeeze(0).T
        filtered_i = F.conv1d(pad_S_i, flip_F_i, padding=0, groups=self.sub_no).squeeze(0).T

        syn_in = filtered_e + filtered_i

        #----- Combine Subunits -----#

        sub_out = torch.zeros(T, self.S//2).cuda()
        
        for s in range(self.S//2):
            sub_idx = -s-1
            leaf_idx = torch.where(self.C_den[sub_idx] == 1)[0]

            if torch.numel(leaf_idx) == 0:
                nonlin_out = torch.tanh(syn_in[:,sub_idx] + self.Theta[sub_idx]) # (T_data,) 
                sub_out[:,sub_idx] = sub_out[:,sub_idx] + nonlin_out
            else:
                leaf_in = sub_out[:,leaf_idx] * torch.exp(self.W_log[leaf_idx]) # (T_data,)
                nonlin_in = syn_in[:,sub_idx] + torch.sum(leaf_in, 1) + self.Theta[sub_idx]# (T_data,)
                nonlin_out = torch.tanh(nonlin_in)
                sub_out[:,sub_idx] = sub_out[:,sub_idx] + nonlin_out
        
        final_voltage = sub_out[:,0]*torch.exp(self.W_log[0]) + self.V_o
        
        return final_voltage, self.mean_u, all_S_u, all_cov_u, all_cov_u_inv, F_e, F_i, self.u_in

        



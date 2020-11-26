import torch
import torch.nn as nn
import torch.nn.functional as F

class gumbel_NS_shGLM(nn.Module):
    def __init__(self, C_den, C_syn_e, C_syn_i, T_syn, syn_basis_no, T_hist,
                hist_basis_no, prop_basis_no):
        super().__init__()

        self.sub_no = C_den.shape[0]
        self.C_den = C_den
        self.C_syn_e = C_syn_e
        self.C_syn_i = C_syn_i
        self.E_no = C_syn_e.shape[1]
        self.syn_basis_no = syn_basis_no
        self.hist_basis_no = hist_basis_no
        self.prop_basis_no = prop_basis_no
        self.T_syn = T_syn
        self.T_hist = T_hist
        
        ### Synapse Parameters ###
        #self.W_s_syn = nn.Parameter(torch.randn(self.sub_no, self.syn_basis_no, 2) * 0.1, requires_grad=True)
        #self.W_ns_syn =  nn.Parameter(torch.randn(self.sub_no, self.syn_basis_no, 2) * 0.1, requires_grad=True)
        #self.Tau_s_syn = nn.Parameter(torch.arange(self.syn_basis_no).float().reshape(-1,1).repeat(1,2) , requires_grad=True)
        #self.Tau_ns_syn = nn.Parameter(torch.arange(self.syn_basis_no).float().reshape(-1,1).repeat(1,2) , requires_grad=True)
        #self.Delta_s_syn = nn.Parameter(torch.rand(self.sub_no, 2), requires_grad=True)
        #self.Delta_ns_syn = nn.Parameter(torch.rand(self.sub_no, 2), requires_grad=True)
        
        self.s_e_kern = nn.Parameter(torch.randn(self.sub_no,  self.T_syn), requires_grad=True)
        self.s_i_kern = nn.Parameter(torch.randn(self.sub_no,  self.T_syn), requires_grad=True)
        #self.ns_e_kern = nn.Parameter(torch.randn(self.sub_no, self.T_syn), requires_grad=True)
        #self.ns_i_kern = nn.Parameter(torch.randn(self.sub_no,  self.T_syn), requires_grad=True)

        ### Ancestor Subunit Parameters ###
        self.W_s_s_sub = nn.Parameter(torch.rand(self.sub_no) , requires_grad=True)
        #self.W_s_ns_sub = nn.Parameter(torch.rand(self.sub_no) , requires_grad=True)
        #self.W_ns_s_sub = nn.Parameter(torch.rand(self.sub_no) , requires_grad=True)
        #self.W_ns_ns_sub = nn.Parameter(torch.rand(self.sub_no) , requires_grad=True)

        ### Subunit Output Parameters ###
        self.V_o = nn.Parameter(torch.randn(1), requires_grad=True)
        self.Theta_s = nn.Parameter(torch.zeros(self.sub_no), requires_grad=True)
        #self.Theta_ns = nn.Parameter(torch.zeros(self.sub_no), requires_grad=True)
        
        self.conv_e = nn.Conv1d(in_channels=self.sub_no, out_channels=self.sub_no,
                               kernel_size=self.T_syn, groups=self.sub_no)
        self.conv_i = nn.Conv1d(in_channels=self.sub_no, out_channels=self.sub_no,
                               kernel_size=self.T_syn, groups=self.sub_no)

    def forward(self, S_e, S_i):
        T_data = S_e.shape[0]
        #print(T_data)

        ############
        # Pre-convolve synapse inputs
        #############
        #syn_s_in = torch.zeros(T_data, self.sub_no).cuda()
        #syn_ns_in = torch.zeros(T_data, self.sub_no).cuda()
        """
        for s in range(self.sub_no):
            
            t_raw = torch.arange(self.T_syn).cuda()

            delta_s_e = torch.exp(self.Delta_s_syn[s,0])
            delta_s_i = torch.exp(self.Delta_s_syn[s,1])
            delta_ns_e = torch.exp(self.Delta_ns_syn[s,0])
            delta_ns_i = torch.exp(self.Delta_ns_syn[s,1])

            t_s_e = t_raw - delta_s_e
            t_s_i = t_raw - delta_s_i
            t_ns_e = t_raw - delta_ns_e
            t_ns_i = t_raw - delta_ns_i
            t_s_e[t_s_e < 0.0] = 0.0
            t_s_i[t_s_i < 0.0] = 0.0
            t_ns_e[t_ns_e < 0.0] = 0.0
            t_ns_i[t_ns_i < 0.0] = 0.0

            full_s_e_kern = torch.zeros(self.T_syn).cuda()
            full_s_i_kern = torch.zeros(self.T_syn).cuda()
            full_ns_e_kern = torch.zeros(self.T_syn).cuda()
            full_ns_i_kern = torch.zeros(self.T_syn).cuda()

            for b in range(self.syn_basis_no):
                tau_s_e = torch.exp(self.Tau_s_syn[b,0])
                tau_s_i = torch.exp(self.Tau_s_syn[b,1])
                tau_ns_e = torch.exp(self.Tau_ns_syn[b,0]) 
                tau_ns_i = torch.exp(self.Tau_ns_syn[b,1])

                t_s_e_tau = t_s_e / tau_s_e
                t_s_i_tau = t_s_i / tau_s_i
                t_ns_e_tau = t_ns_e / tau_ns_e
                t_ns_i_tau = t_ns_i / tau_ns_i

                part_s_e_kern = t_s_e_tau * torch.exp(-t_s_e_tau)
                part_s_i_kern = t_s_i_tau * torch.exp(-t_s_i_tau)
                part_ns_e_kern = t_ns_e_tau * torch.exp(-t_ns_e_tau)
                part_ns_i_kern = t_ns_i_tau * torch.exp(-t_ns_i_tau)
                
                full_s_e_kern = full_s_e_kern + part_s_e_kern * self.W_s_syn[s,b,0]
                full_s_i_kern = full_s_i_kern + part_s_i_kern * self.W_s_syn[s,b,1]
                full_ns_e_kern = full_ns_e_kern + part_ns_e_kern * self.W_ns_syn[s,b,0]
                full_ns_i_kern = full_ns_i_kern + part_ns_i_kern * self.W_ns_syn[s,b,1]

            in_e = torch.matmul(S_e, self.C_syn_e.T[:,s])
            in_i = torch.matmul(S_i, self.C_syn_i.T[:,s])
            pad_in_e = torch.zeros(T_data + self.T_syn - 1).cuda()
            pad_in_i = torch.zeros(T_data + self.T_syn - 1).cuda()
            pad_in_e[-T_data:] = pad_in_e[-T_data:] + in_e
            pad_in_i[-T_data:] = pad_in_i[-T_data:] + in_i
            pad_in_e = pad_in_e.reshape(1,1,-1)
            pad_in_i = pad_in_i.reshape(1,1,-1)

            full_s_e_kern = torch.flip(full_s_e_kern, [0])
            full_s_i_kern = torch.flip(full_s_i_kern, [0])
            full_ns_e_kern = torch.flip(full_ns_e_kern, [0])
            full_ns_i_kern = torch.flip(full_ns_i_kern, [0])

            full_s_e_kern = full_s_e_kern.reshape(1,1,-1)
            full_s_i_kern = full_s_i_kern.reshape(1,1,-1)
            full_ns_e_kern = full_ns_e_kern.reshape(1,1,-1)
            full_ns_i_kern = full_ns_i_kern.reshape(1,1,-1)
            
            
            full_s_e_kern = self.s_e_kern[s].reshape(1,1,-1)
            full_s_i_kern = self.s_i_kern[s].reshape(1,1,-1)
            #full_ns_e_kern = self.ns_e_kern[s].reshape(1,1,-1)
            #full_ns_i_kern = self.ns_i_kern[s].reshape(1,1,-1)
            
            in_e = torch.matmul(S_e, self.C_syn_e.T[:,s])
            in_i = torch.matmul(S_i, self.C_syn_i.T[:,s])
            pad_in_e = torch.zeros(T_data + self.T_syn - 1).cuda()
            pad_in_i = torch.zeros(T_data + self.T_syn - 1).cuda()
            pad_in_e[-T_data:] = pad_in_e[-T_data:] + in_e
            pad_in_i[-T_data:] = pad_in_i[-T_data:] + in_i
            pad_in_e = pad_in_e.reshape(1,1,-1)
            pad_in_i = pad_in_i.reshape(1,1,-1)

            filtered_s_e = F.conv1d(pad_in_e, full_s_e_kern, padding=0).squeeze(1).T
            filtered_s_i = F.conv1d(pad_in_i, full_s_i_kern, padding=0).squeeze(1).T
            #filtered_ns_e = F.conv1d(pad_in_e, full_ns_e_kern, padding=0).squeeze(1).T
            #filtered_ns_i = F.conv1d(pad_in_i, full_ns_i_kern, padding=0).squeeze(1).T

            syn_s_in[:,s] = syn_s_in[:,s] + filtered_s_e.flatten() + filtered_s_i.flatten()
            #syn_ns_in[:,s] = syn_ns_in[:,s] + filtered_ns_e.flatten() + filtered_ns_i.flatten()
        """
        #############
        # Solve for X_t, Y_t, Z_t through time
        ###########

        #Y_s_pad = torch.zeros(T_data+1, self.sub_no).cuda() ### NONscaled ancestor subunit inputs to spiking subunit
        #Y_ns_pad = torch.zeros(T_data+1, self.sub_no).cuda() ### NONScaled ancestor subunit inputs to non-spiking subunit

        """
        for t in range(T_data):
            Y_s_slice = Y_s_pad[t].clone()
            Y_ns_slice = Y_ns_pad[t].clone()
            
            raw_s_s_prop = torch.matmul(Y_s_slice*self.W_s_s_sub, self.C_den[:,:].T) # sub_no
            raw_s_ns_prop = torch.matmul(Y_s_slice*self.W_s_ns_sub, self.C_den[:,:].T) # sub_no
            raw_ns_s_prop = torch.matmul(Y_ns_slice*self.W_ns_s_sub, self.C_den[:,:].T) # sub_no
            raw_ns_ns_prop = torch.matmul(Y_ns_slice*self.W_ns_ns_sub, self.C_den.T) # sub_no
                
            Y_s_slice_out= torch.sigmoid(syn_s_in[t,:] + raw_s_s_prop + raw_ns_s_prop + self.Theta_s)
            Y_ns_slice_out = torch.sigmoid(syn_ns_in[t,:] + raw_s_ns_prop + raw_ns_ns_prop + self.Theta_ns)
            
            Y_s_pad[t+1] =  Y_s_slice_out
            Y_ns_pad[t+1] =  Y_ns_slice_out
        """
        
        syn_e = torch.matmul(S_e, self.C_syn_e.T)
        syn_i = torch.matmul(S_i, self.C_syn_i.T)
        syn_e_pad = torch.vstack((torch.zeros( self.T_syn - 1, self.sub_no).cuda() , syn_e))
        syn_i_pad = torch.vstack((torch.zeros( self.T_syn - 1, self.sub_no).cuda() , syn_i))
        syn_e_in = self.conv_e(syn_e_pad.T.reshape(1,self.sub_no,-1))
        syn_i_in = self.conv_i(syn_i_pad.T.reshape(1,self.sub_no,-1))
        #syn_e_in = F.conv1d(syn_e_pad.T.reshape(1,self.sub_no,-1), self.s_e_kern.reshape(1,))
        
        syn_in = syn_e_in[0].T + syn_i_in[0].T
        
        Y_s_pad = syn_in + torch.matmul(syn_in * torch.exp(self.W_s_s_sub), self.C_den.T)
        #Y_ns_pad = syn_s_in + torch.matmul(syn_ns_in * self.W_s_ns_sub, self.C_den.T) + torch.matmul(syn_ns_in*self.W_ns_ns_sub, self.C_den.T)
        
        #final_voltage = Y_ns_pad[:,0]*self.W_ns_ns_sub[0] + Y_s_pad[:,0]*self.W_s_s_sub[0] + self.V_o
        final_voltage = Y_s_pad[:,0]*torch.exp(self.W_s_s_sub[0])
        #final_Y_ns = Y_ns_pad[:,1:]
        final_Y_s = Y_s_pad[:,:]
        #print(Y_s_pad.shape)
        #print(Y_ns_pad.shape)
        
        return final_voltage, final_Y_s, final_Y_s

            

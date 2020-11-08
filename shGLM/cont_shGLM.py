import torch
import torch.nn as nn
import torch.nn.functional as F

class cont_shGLM(nn.Module):
    def __init__(self, C_den, C_syn_e, C_syn_i, E_no, T_syn, syn_basis_no,
                spike_status, T_hist, hist_basis_no):
        super().__init__()

        self.sub_no = C_den.shape[0]
        self.C_den = C_den
        self.C_syn_e = C_syn_e
        self.C_syn_i = C_syn_i
        self.E_no = E_no
        self.syn_basis_no = syn_basis_no
        self.hist_basis_no = hist_basis_no
        self.spike_status = spike_status
        self.T_syn = T_syn
        self.T_hist = T_hist

        ### Synaptic Parameters ###
        self.V_o = nn.Parameter(torch.ones(1)*(-69), requires_grad=True)
        self.K_syn_raw = torch.ones(self.sub_no, self.syn_basis_no, 3)
        self.K_syn_raw[:,:,0] = 0.04
        self.K_syn_raw[:,:,1] = -0.02
        self.K_syn_raw[:,:,2] = 0.025
        self.K_syn = nn.Parameter(self.K_syn_raw, requires_grad=True)
        self.tau_syn = nn.Parameter(torch.arange(1.6, 1.6+self.syn_basis_no).reshape(-1,1).repeat(1,3),
                                            requires_grad=True)
        self.delta_syn = nn.Parameter(torch.ones(self.sub_no, 2) * 2, requires_grad=True)
        self.W_sub = nn.Parameter(torch.ones(self.sub_no) * 1 , requires_grad=True)
        self.theta_syn = nn.Parameter(torch.zeros(self.sub_no), requires_grad=True)

        ### Spiking Parameters ###
        self.theta_spike = nn.Parameter(torch.ones(self.sub_no - 1)*(-150), requires_grad=True)
        self.W_spike = nn.Parameter(torch.ones(self.sub_no - 1)*100, requires_grad=True)
        self.tau_hist = nn.Parameter(torch.arange(0.5, 0.5+self.hist_basis_no*0.5,step=0.5),
                                        requires_grad=True)
        self.K_spike = nn.Parameter(torch.ones(self.sub_no - 1, self.hist_basis_no) * (-0.04), requires_grad=True)

    def forward(self, S_e, S_i):
        T_data = S_e.shape[0]

        ############
        # Pre-convolve synapse inputs
        #############
        syn_in = torch.zeros(T_data, self.sub_no).to("cuda")

        for s in range(self.sub_no):
            t_e = torch.arange(self.T_syn).to("cuda")
            t_i = torch.arange(self.T_syn).to("cuda")
            delta_e = torch.exp(self.delta_syn[s,0])
            delta_i = torch.exp(self.delta_syn[s,1])
            t_e = t_e - delta_e
            t_i = t_i - delta_i
            t_e[t_e < 0.0] = 0.0
            t_i[t_i < 0.0] = 0.0

            full_e_kern = torch.zeros(self.T_syn).to("cuda")
            full_i_kern = torch.zeros(self.T_syn).to("cuda")

            for b in range(self.syn_basis_no):
                tau_e = torch.exp(self.tau_syn[b,0])
                tau_i = torch.exp(self.tau_syn[b,1])
                t_e_tau = t_e / tau_e
                t_i_tau = t_i / tau_i
                part_e_kern = t_e_tau * torch.exp(-t_e_tau)
                part_i_kern = t_i_tau * torch.exp(-t_i_tau)
                full_e_kern = full_e_kern + part_e_kern * self.K_syn[s,b,0]
                full_i_kern = full_i_kern + part_i_kern * self.K_syn[s,b,1]

            in_e = torch.matmul(S_e, self.C_syn_e.T[:,s])
            in_i = torch.matmul(S_i, self.C_syn_i.T[:,s])
            pad_in_e = torch.zeros(T_data + self.T_syn - 1).cuda(0)
            pad_in_i = torch.zeros(T_data + self.T_syn - 1).to("cuda")
            pad_in_e[-T_data:] = pad_in_e[-T_data:] + in_e
            pad_in_i[-T_data:] = pad_in_i[-T_data:] + in_i
            
            pad_in_e = pad_in_e.reshape(1,1,-1)
            pad_in_i = pad_in_i.reshape(1,1,-1)
            full_e_kern = torch.flip(full_e_kern, [0])
            full_i_kern = torch.flip(full_i_kern, [0])
            full_e_kern = full_e_kern.reshape(1,1,-1)
            full_i_kern = full_i_kern.reshape(1,1,-1)

            filtered_e = F.conv1d(pad_in_e, full_e_kern, padding=0)
            filtered_i = F.conv1d(pad_in_i, full_i_kern, padding=0)
            filtered_e = filtered_e.squeeze(1).T
            filtered_i = filtered_i.squeeze(1).T
            syn_in[:,s] = syn_in[:,s] + filtered_e.flatten() + filtered_i.flatten()

        #############
        # Solve for X_t, Y_t, Z_t through time
        ###########

        X = torch.empty(T_data, self.sub_no).to("cuda") ### after first nonlinearity, before subunit scalar
        Z_pad = torch.zeros(T_data + self.T_hist, self.sub_no - 1).to("cuda") ### spike trains
        Y = torch.zeros(T_data+1, self.sub_no).to("cuda") ### after subunit scalar

        hist_kern = torch.zeros(self.sub_no - 1, self.T_hist).to("cuda")
        for s in range(self.sub_no - 1):
            t = torch.arange(self.T_hist).to("cuda")
            for b in range(self.hist_basis_no):
                tau = torch.exp(self.tau_hist[b])
                t_tau = t / tau
                hist_kern[s,:] = hist_kern[s,:] + t_tau * torch.exp(-t_tau) * self.K_spike[s,b]
        hist_kern = torch.flip(hist_kern, [1]).reshape(self.sub_no - 1, 1, -1)

        ancest_kern = torch.zeros(self.sub_no, self.T_hist).to("cuda")
        for s in range(self.sub_no):
            t = torch.arange(self.T_hist).to("cuda")
            for b in range(self.syn_basis_no):
                tau = torch.exp(self.tau_syn[b,2])
                t_tau = t / tau
                ancest_kern[s,:] = ancest_kern[s,:] + t_tau * torch.exp(-t_tau) * self.K_syn[s,b,2]
        ancest_kern = torch.flip(ancest_kern, [1]).reshape(self.sub_no, 1, -1)
        
        spike_hist = Z_pad[:self.T_hist , :].T
        for t in range(T_data):
            spike_ancest = torch.matmul(self.C_den[:,1:], spike_hist)
            spike_hist = Z_pad[t : t+self.T_hist , :].T.reshape(1, self.sub_no-1, -1)
            spike_ancest = spike_ancest.reshape(1, self.sub_no, -1)
            raw_ancest = torch.matmul(self.C_den, Y[t-1])

            filtered_ancest = F.conv1d(spike_ancest, ancest_kern, groups=self.sub_no)
            filtered_hist = F.conv1d(spike_hist, hist_kern, groups=self.sub_no-1)
            filtered_ancest = filtered_ancest.flatten()
            filtered_hist = filtered_hist.flatten()
       
            X_in = syn_in[t] + filtered_ancest + raw_ancest + self.theta_syn
            X_in[1:] = X_in[1:] + filtered_hist
            X_out = torch.sigmoid(X_in)
            X[t] = X_out
            Y[t+1] = X_out * self.W_sub

            Z_mu = X_out[1:] * self.W_spike + self.theta_spike
            Z_in = Z_mu + torch.randn(self.sub_no-1).cuda() * 100
            Z_out =  torch.sigmoid(Z_in)
            Z_pad[t+self.T_hist] = Z_out
            
            spike_hist = torch.cat((spike_hist.reshape(-1,self.T_hist)[:,1:], Z_out.reshape(-1,1)),1)

        final_voltage = Y[1:,0] + self.V_o
        final_Y = Y[1:,1:]
        final_Z = Z_pad[self.T_hist:,:]

        return final_voltage, final_Y, final_Z

            

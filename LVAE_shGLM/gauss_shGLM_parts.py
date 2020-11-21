import torch
from torch import nn
from torch.nn import functional as F

class Leaf_Integ(nn.Module):
    def __init__(self, sub_no, syn_basis_no,
                T_hist, hist_basis_no, fix_var, theta_spike_init, W_spike_init):
        super().__init__()

        self.syn_basis_no = syn_basis_no
        self.T_hist = T_hist
        self.hist_basis_no = hist_basis_no
        self.sub_no = sub_no
        self.fix_var = fix_var
        self.fix_prec = fix_var ** (-1)
        self.theta_spike_init = theta_spike_init
        self.W_spike_init = W_spike_init

        ### Synaptic Parameters ###
        self.W_sub = nn.Parameter(torch.ones(self.sub_no) * 0.25 , requires_grad=True)
        self.theta_syn = nn.Parameter(torch.zeros(self.sub_no), requires_grad=True)
        
        ### Spiking Parameters ###
        self.theta_spike = nn.Parameter(torch.ones(self.sub_no)*(self.theta_spike_init), requires_grad=True) ###
        self.W_spike = nn.Parameter(torch.ones(self.sub_no)*self.W_spike_init, requires_grad=True) ###
        self.tau_hist = nn.Parameter(torch.arange(0.5, 0.5+self.hist_basis_no*0.5,step=0.5),
                                        requires_grad=True)
        self.K_hist = nn.Parameter(torch.zeros(self.sub_no, self.hist_basis_no), requires_grad=True)
        self.delta_hist = nn.Parameter(torch.zeros(self.sub_no), requires_grad=True)

    def forward(self, S_conv, up_mu_Z = None):
        T_data = S_conv.shape[0]

        X = torch.zeros(T_data, self.sub_no).cuda() ### after first nonlinearity, before subunit scalar
        Z_pad = torch.zeros(T_data + self.T_hist, self.sub_no).cuda() ### spike train means 
        Y = torch.zeros(T_data+1, self.sub_no).cuda() ### after subunit scalar
        mu_Z = torch.empty(T_data, self.sub_no).cuda()
        down_mu_Z = torch.empty(T_data, self.sub_no).cuda()

        hist_kern = torch.zeros(self.sub_no, self.T_hist).cuda()
        for s in range(self.sub_no):
            t = torch.arange(self.T_hist).cuda()
            delta = self.delta_hist[s]
            t = t - delta
            t[t < 0.0] = 0.0
            for b in range(self.hist_basis_no):
                tau = torch.exp(self.tau_hist[b])
                t_tau = t / tau
                hist_kern[s,:] = hist_kern[s,:] + t_tau * torch.exp(-t_tau) * self.K_hist[s,b]
        hist_kern = torch.flip(hist_kern, [1]).reshape(self.sub_no, 1, -1)

        spike_hist = Z_pad[:self.T_hist , :].T
        for t in range(T_data):
            spike_hist = spike_hist.reshape(1, self.sub_no, -1)
            filtered_hist = F.conv1d(spike_hist, hist_kern, groups=self.sub_no)
            filtered_hist = filtered_hist.flatten()
            X_in = S_conv[t] + filtered_hist + self.theta_syn
            X[t] = X_in
            X_out = torch.sigmoid(X_in)
            Y[t+1] = X_out * self.W_sub
            down_mu_Z_t = X_out * self.W_spike + self.theta_spike
            
            if up_mu_Z == None:
                mu_Z_t = down_mu_Z_t
            else:
                mu_Z_t = (up_mu_Z[t]*self.fix_prec + down_mu_Z_t*self.fix_prec) / (2 * self.fix_prec)

            Z_in = mu_Z_t + torch.randn(self.sub_no).cuda() * self.fix_var**(0.5)
            Z_out = torch.sigmoid(Z_in)
            Z_pad[t+self.T_hist] = Z_out
            spike_hist = torch.cat((spike_hist.reshape(-1,self.T_hist)[:,1:], Z_out.reshape(-1,1)),1)
            mu_Z[t] = mu_Z_t
            down_mu_Z[t] = down_mu_Z_t
        
        #print("LEAF")
        #print(X[300:320])
        #print(S_conv[300:320])
        
        final_Y = Y[1:,:]
        final_Z = Z_pad[self.T_hist:,:]

        return final_Y, final_Z, mu_Z, down_mu_Z

class Middle_Integ(nn.Module):
    def __init__(self, C_den, syn_basis_no,
                T_hist, hist_basis_no, fix_var, theta_spike_init, W_spike_init):
        super().__init__()

        self.C_den = C_den
        self.syn_basis_no = syn_basis_no
        self.T_hist = T_hist
        self.hist_basis_no = hist_basis_no
        self.sub_no = part_C_den.shape[0]
        self.fix_var = fix_var
        self.fix_prec = fix_var ** (-1)
        self.theta_spike_init = theta_spike_init
        self.W_spike_init = W_spike_init

        ### Synaptic Parameters ###
        self.W_sub = nn.Parameter(torch.ones(self.sub_no) * 0.25 , requires_grad=True)
        self.theta_syn = nn.Parameter(torch.zeros(self.sub_no), requires_grad=True)
        self.K_spike = nn.Parameter(torch.ones(self.sub_no, self.syn_basis_no) * 0.01, requires_grad=True)
        self.tau_spike = nn.Parameter(torch.arange(1.6, 1.6+self.syn_basis_no))
        self.delta_spike = nn.Parameter(torch.zeros(self.sub_no), requires_grad=True)

        ### Spiking Parameters ###
        self.theta_spike = nn.Parameter(torch.ones(self.sub_no)*(self.theta_spike_init), requires_grad=True) ###
        self.W_spike = nn.Parameter(torch.ones(self.sub_no)*self.W_spike_init, requires_grad=True) ###
        self.tau_hist = nn.Parameter(torch.arange(0.5, 0.5+self.hist_basis_no*0.5,step=0.5),
                                        requires_grad=True)
        self.K_hist = nn.Parameter(torch.zeros(self.sub_no, self.hist_basis_no), requires_grad=True)
        self.delta_hist = nn.Parameter(torch.zeros(self.sub_no), requires_grad=True)

    def forward(self, S_conv, Y_ancest, Z_ancest, up_mu_Z = None):
        T_data = S_conv.shape[0]

        X = torch.empty(T_data, self.sub_no).cuda() ### after first nonlinearity, before subunit scalar
        Z_pad = torch.zeros(T_data + self.T_hist, self.sub_no).cuda() ### spike train means 
        Y = torch.zeros(T_data+1, self.sub_no).cuda() ### after subunit scalar
        Z_ancest_pad = torch.zeros(T_data + self.T_hist, Z_ancest.shape[1]).cuda()
        Z_ancest_pad[-T_data:] = Z_ancest_pad[-T_data:] + Z_ancest
        mu_Z = torch.empty(T_data, self.sub_no).cuda()
        down_mu_Z = torch.empty(T_data, self.sub_no).cuda()

        hist_kern = torch.zeros(self.sub_no, self.T_hist).cuda()
        for s in range(self.sub_no):
            t = torch.arange(self.T_hist).cuda()
            delta = self.delta_hist[s]
            t = t - delta
            t[t < 0.0] = 0.0
            for b in range(self.hist_basis_no):
                tau = torch.exp(self.tau_hist[b])
                t_tau = t / tau
                hist_kern[s,:] = hist_kern[s,:] + t_tau * torch.exp(-t_tau) * self.K_hist[s,b]
        hist_kern = torch.flip(hist_kern, [1]).reshape(self.sub_no, 1, -1)

        ancest_kern = torch.zeros(self.sub_no, self.T_hist).cuda()
        for s in range(self.sub_no):
            t = torch.arange(self.T_hist).cuda()
            delta = self.delta_spike[s]
            t = t - delta
            t[t < 0.0] = 0.0
            for b in range(self.syn_basis_no):
                tau = torch.exp(self.tau_spike[b])
                t_tau = t / tau
                ancest_kern[s,:] = ancest_kern[s,:] + t_tau * torch.exp(-t_tau) * self.K_spike[s,b]
        ancest_kern = torch.flip(ancest_kern, [1]).reshape(self.sub_no, 1, -1)

        spike_hist = Z_pad[:self.T_hist , :].T
        for t in range(T_data):
            spike_ancest = torch.matmul(self.C_den, Z_ancest_pad[t:t+self.T_hist,:].T)
            spike_hist = spike_hist.reshape(1, self.sub_no, -1)
            spike_ancest = spike_ancest.reshape(1, self.sub_no, -1)
            raw_ancest = torch.matmul(self.C_den, Y_ancest[t])


            filtered_ancest = F.conv1d(spike_ancest, ancest_kern, groups=self.sub_no)
            filtered_hist = F.conv1d(spike_hist, hist_kern, groups=self.sub_no)
            filtered_hist = filtered_hist.flatten()
            filtered_ancest = filtered_ancest.flatten()

            X_in = S_conv[t] + filtered_hist + self.theta_syn + filtered_ancest + raw_ancest
            X_out = torch.sigmoid(X_in)
            X[t] = X_in
            Y[t+1] = X_out * self.W_sub
            down_mu_Z_t = X_out * self.W_spike + self.theta_spike
            
            if up_mu_Z == None:
                mu_Z_t = down_mu_Z_t
            else:
                mu_Z_t = (up_mu_Z[t]*self.fix_prec + down_mu_Z_t*self.fix_prec) / (2 * self.fix_prec)

            Z_in = mu_Z_t + torch.randn(self.sub_no).cuda() * self.fix_var**(0.5)
            Z_out = torch.sigmoid(Z_in)
            Z_pad[t+self.T_hist] = Z_out
            spike_hist = torch.cat((spike_hist.reshape(-1,self.T_hist)[:,1:], Z_out.reshape(-1,1)),1)
            mu_Z[t] = mu_Z_t
            down_mu_Z[t] = down_mu_Z_t

        final_Y = Y[1:,:]
        final_Z = Z_pad[self.T_hist:,:]

        return final_Y, final_Z, mu_Z, down_mu_Z

class Root_Integ(nn.Module):
    def __init__(self, C_den, syn_basis_no, T_hist):
        super().__init__()

        self.C_den = C_den
        self.syn_basis_no = syn_basis_no
        self.sub_no = C_den.shape[0]

        ### Synaptic Parameters ###
        self.V_o = nn.Parameter(torch.ones(1)*(-69), requires_grad=True)
        self.W_sub = nn.Parameter(torch.ones(self.sub_no) * 0.25 , requires_grad=True)
        self.theta_syn = nn.Parameter(torch.zeros(self.sub_no), requires_grad=True)
        self.K_spike = nn.Parameter(torch.ones(self.sub_no, self.syn_basis_no) * 0.01, requires_grad=True)
        self.tau_spike = nn.Parameter(torch.arange(1.6, 1.6+self.syn_basis_no))
        self.delta_spike = nn.Parameter(torch.zeros(self.sub_no), requires_grad=True)
        self.T_hist = T_hist

    def forward(self, S_conv, Y_ancest, Z_ancest):
        T_data = S_conv.shape[0]

        X = torch.empty(T_data, self.sub_no).cuda() ### after first nonlinearity, before subunit scalar
        Z_ancest_pad = torch.zeros(T_data + self.T_hist, Z_ancest.shape[1]).cuda()
        Z_ancest_pad[-T_data:] = Z_ancest_pad[-T_data:] + Z_ancest
        Y = torch.zeros(T_data+1, self.sub_no).cuda() ### after subunit scalar

        ancest_kern = torch.zeros(self.sub_no, self.T_hist).cuda()
        for s in range(self.sub_no):
            t = torch.arange(self.T_hist).cuda()
            delta = self.delta_spike[s]
            t = t - delta
            t[t < 0.0] = 0.0
            for b in range(self.syn_basis_no):
                tau = torch.exp(self.tau_spike[b])
                t_tau = t / tau
                ancest_kern[s,:] = ancest_kern[s,:] + t_tau * torch.exp(-t_tau) * self.K_spike[s,b]
        ancest_kern = torch.flip(ancest_kern, [1]).reshape(self.sub_no, 1, -1)

        for t in range(T_data):
            spike_ancest = torch.matmul(self.C_den, Z_ancest_pad[t:t+self.T_hist].T)
            spike_ancest = spike_ancest.reshape(1, self.sub_no, -1)
            raw_ancest = torch.matmul(self.C_den, Y_ancest[t])

            filtered_ancest = F.conv1d(spike_ancest, ancest_kern, groups=self.sub_no)
            filtered_ancest = filtered_ancest.flatten()

            X_in = S_conv[t] + self.theta_syn + filtered_ancest + raw_ancest
            X_out = torch.sigmoid(X_in)
            X[t] = X_in
            Y[t+1] = X_out * self.W_sub
        
        #print("ROOT")
        #print(X[300:320])
        #print(S_conv[300:320])
        #print(filtered_ancest)
        #print(raw_ancest)
        #print(self.theta_syn)
        final_Y = Y[1:,:] + self.V_o

        return final_Y

class Spike_Convolve(nn.Module):
    def __init__(self, C_syn_e, C_syn_i, T_syn, syn_basis_no):
        super().__init__()

        self.sub_no = C_syn_e.shape[0]
        self.C_syn_e = C_syn_e
        self.C_syn_i = C_syn_i
        self.syn_basis_no = syn_basis_no
        self.T_syn = T_syn

        ### Synaptic Parameters ###
        self.K_syn_raw = torch.ones(self.sub_no, self.syn_basis_no, 2)
        self.K_syn_raw[:,:,0] = 0.02
        self.K_syn_raw[:,:,1] = -0.02
        self.K_syn = nn.Parameter(self.K_syn_raw, requires_grad=True)
        self.tau_syn = nn.Parameter(torch.arange(1.6, 1.6+self.syn_basis_no).reshape(-1,1).repeat(1,2),
                                            requires_grad=True)
        self.delta_syn = nn.Parameter(torch.zeros(self.sub_no, 2), requires_grad=True)

    def forward(self, S_e, S_i):
        T_data = S_e.shape[0]

        syn_in = torch.zeros(T_data, self.sub_no).cuda()

        for s in range(self.sub_no):
            t_e = torch.arange(self.T_syn).cuda()
            t_i = torch.arange(self.T_syn).cuda()
            delta_e = self.delta_syn[s,0]
            delta_i = self.delta_syn[s,1]
            t_e = t_e - delta_e
            t_i = t_i - delta_i
            t_e[t_e < 0.0] = 0.0
            t_i[t_i < 0.0] = 0.0

            full_e_kern = torch.zeros(self.T_syn).cuda()
            full_i_kern = torch.zeros(self.T_syn).cuda()

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
            pad_in_e = torch.zeros(T_data + self.T_syn - 1).cuda()
            pad_in_i = torch.zeros(T_data + self.T_syn - 1).cuda()
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
        
        #print("SPIKE")
        #print(self.K_syn)
        return syn_in

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, latent_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.PReLU(),
            nn.Linear(out_dim, out_dim),
            nn.PReLU())

        self.mean_layer = nn.Sequential(
            nn.Linear(out_dim, latent_dim))

    def encode(self, x, outmeanvar=False):
        h = self.net(x)
        mu = self.mean_layer(h)
        return h, mu

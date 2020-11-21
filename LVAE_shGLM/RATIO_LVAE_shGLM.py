import torch
import torch.nn as nn
import torch.nn.functional as F

class RATIO_LVAE_shGLM(nn.Module):
    def __init__(self, C_den, C_syn_e, C_syn_i, E_no, T_syn, syn_basis_no,
                T_hist, hist_basis_no, T_enc, hid_dim, temp):
        super().__init__()

        self.sub_no = C_den.shape[0]
        self.C_den = C_den
        self.C_syn_e = C_syn_e
        self.C_syn_i = C_syn_i
        self.E_no = E_no
        self.syn_basis_no = syn_basis_no
        self.hist_basis_no = hist_basis_no
        self.T_syn = T_syn
        self.T_hist = T_hist
        self.T_enc = T_enc
        self.hid_dim = hid_dim
        self.temp = temp

        ### Synaptic Parameters ###
        self.V_o = nn.Parameter(torch.ones(1)*(-69), requires_grad=True)
        self.K_syn_raw = torch.ones(self.sub_no, self.syn_basis_no, 3)
        self.K_syn_raw[:,:,0] = 0.04
        self.K_syn_raw[:,:,1] = -0.02
        self.K_syn_raw[:,:,2] = 0.02
        self.K_syn = nn.Parameter(self.K_syn_raw, requires_grad=True)
        self.tau_syn = nn.Parameter(torch.arange(1.6, 1.6+self.syn_basis_no).reshape(-1,1).repeat(1,3),
                                            requires_grad=True)
        self.delta_syn = nn.Parameter(torch.ones(self.sub_no, 2) * 2, requires_grad=True)
        self.W_sub = nn.Parameter(torch.ones(self.sub_no) * 1 , requires_grad=True)
        self.theta_syn = nn.Parameter(torch.zeros(self.sub_no), requires_grad=True)

        ### Spiking Parameters ###
        self.theta_spike = nn.Parameter(torch.ones(self.sub_no - 1)*(0), requires_grad=True)
        self.W_spike = nn.Parameter(torch.ones(self.sub_no - 1)*(1), requires_grad=True)
        self.tau_hist = nn.Parameter(torch.arange(0.5, 0.5+self.hist_basis_no*0.5,step=0.5),
                                        requires_grad=True)
        self.K_spike = nn.Parameter(torch.ones(self.sub_no - 1, self.hist_basis_no)*(-1)
                                    , requires_grad=True)

        ### Encoding Model ###
        self.MLP_enc = nn.Sequential(nn.Linear(self.T_enc*2-1, self.hid_dim),
                                nn.PReLU(),
                                nn.Linear(self.hid_dim, self.hid_dim),
                                nn.PReLU(),
                                nn.Linear(self.hid_dim, self.hid_dim),
                                nn.PReLU(),
                                nn.Linear(self.hid_dim, self.sub_no-1))
        self.enc_bias = nn.Parameter(torch.ones(self.sub_no-1) * 0, requires_grad=True)

    def spike_convolve(self, S_e, S_i):
        T_data = S_e.shape[0]
        syn_in = torch.zeros(T_data, self.sub_no).cuda()

        for s in range(self.sub_no):
            t_e = torch.arange(self.T_syn).cuda()
            t_i = torch.arange(self.T_syn).cuda()
            delta_e = torch.exp(self.delta_syn[s,0])
            delta_i = torch.exp(self.delta_syn[s,1])
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

        return syn_in
    
    def encode(self, V_in, S_e, S_i):
        T_data = V_in.shape[0]
        
        S_conv = self.spike_convolve(S_e, S_i)
        V_in_pad = torch.zeros(T_data + 2*self.T_enc - 1).cuda()
        V_in_pad[self.T_enc-1:self.T_enc+T_data-1] = V_in_pad[self.T_enc-1:self.T_enc+T_data-1] + V_in
        NN_out = torch.zeros(T_data, self.sub_no-1).cuda()

        for t in range(T_data):
            NN_t = self.MLP_enc(V_in_pad[t : t+2*self.T_enc-1])
            NN_out[t] = NN_out[t] + NN_t

        posterior_probs_ratios = torch.sigmoid(NN_out + S_conv[:,1:] + self.enc_bias)
        return posterior_probs_ratios

    def decode(self, S_e, S_i, posterior_probs_ratios=None):
        T_data = S_e.shape[0]
        
        S_conv = self.spike_convolve(S_e, S_i)

        hist_kern = torch.zeros(self.sub_no - 1, self.T_hist).cuda()
        for s in range(self.sub_no - 1):
            t = torch.arange(self.T_hist).cuda()
            for b in range(self.hist_basis_no):
                tau = torch.exp(self.tau_hist[b])
                t_tau = t / tau
                hist_kern[s,:] = hist_kern[s,:] + t_tau * torch.exp(-t_tau) * self.K_spike[s,b]
        hist_kern = torch.flip(hist_kern, [1]).reshape(self.sub_no - 1, 1, -1)

        ancest_kern = torch.zeros(self.sub_no, self.T_hist).cuda()
        for s in range(self.sub_no):
            t = torch.arange(self.T_hist).cuda()
            for b in range(self.syn_basis_no):
                tau = torch.exp(self.tau_syn[b,2])
                t_tau = t / tau
                ancest_kern[s,:] = ancest_kern[s,:] + t_tau * torch.exp(-t_tau) * self.K_syn[s,b,2]
        ancest_kern = torch.flip(ancest_kern, [1]).reshape(self.sub_no, 1, -1)

        ### CHANGES DEPENDING ON INFERENCE OR GENERATION
        if posterior_probs_ratios == None:
            X = torch.zeros(T_data, self.sub_no).cuda() ### after first nonlinearity, before subunit scalar
            Y = torch.zeros(T_data+1, self.sub_no).cuda() ### after subunit scalar
            Z_ratios_out = torch.zeros(T_data, self.sub_no-1).cuda()
            Z_samples = torch.zeros(T_data + self.T_hist, self.sub_no - 1).cuda() ### spike trains

            spike_hist = Z_samples[:self.T_hist , :].T
            for t in range(T_data):
                spike_ancest = torch.matmul(self.C_den[:,1:], spike_hist)
                spike_hist = spike_hist.reshape(1, self.sub_no-1, -1)
                spike_ancest = spike_ancest.reshape(1, self.sub_no, -1)
                raw_ancest = torch.matmul(self.C_den, Y[t])

                filtered_ancest = F.conv1d(spike_ancest, ancest_kern, groups=self.sub_no)
                filtered_hist = F.conv1d(spike_hist, hist_kern, groups=self.sub_no-1)
                filtered_ancest = filtered_ancest.flatten()
                filtered_hist = filtered_hist.flatten()
        
                X_in = S_conv[t] + filtered_ancest + raw_ancest + self.theta_syn
                X_in[1:] = X_in[1:] + filtered_hist
                X_out = torch.sigmoid(X_in)
                X[t] = X_out
                Y[t+1] = X_out * self.W_sub
       
                Z_ratios = torch.sigmoid(X_out[1:] * self.W_spike + self.theta_spike)
                uniform = torch.rand(self.sub_no-1).cuda()
                logistic = torch.log(uniform +1e-10) - torch.log(1-uniform +1e-10)
                Z_pre = (torch.log(Z_ratios +1e-10) + logistic) / self.temp
                Z_post = torch.sigmoid(Z_pre)
                
                Z_ratios_out[t] = Z_ratios_out[t] + Z_ratios
                Z_samples[t+self.T_hist] = Z_samples[t+self.T_hist] + Z_post
                spike_hist = torch.cat((spike_hist.reshape(-1,self.T_hist)[:,1:], Z_post.reshape(-1,1)),1)

            final_voltage = Y[1:,0] + self.V_o
            final_Y = Y[1:,1:]
            final_Z_samples = Z_samples[self.T_hist:,:]
            
            return final_voltage, final_Y, final_Z_samples, Z_ratios_out

        else:
            X = torch.zeros(T_data, self.sub_no).cuda() ### after first nonlinearity, before subunit scalar
            Y = torch.zeros(T_data+1, self.sub_no).cuda() ### after subunit scalar
            Z_prior_ratios_out = torch.zeros(T_data, self.sub_no-1).cuda()
            Z_post_ratios_out = torch.zeros(T_data, self.sub_no-1).cuda()
            Z_samples = torch.zeros(T_data + self.T_hist, self.sub_no - 1).cuda() ### spike trains

            spike_hist = Z_samples[:self.T_hist , :].T
            for t in range(T_data):
                spike_ancest = torch.matmul(self.C_den[:,1:], spike_hist)
                spike_hist = spike_hist.reshape(1, self.sub_no-1, -1)
                spike_ancest = spike_ancest.reshape(1, self.sub_no, -1)
                raw_ancest = torch.matmul(self.C_den, Y[t])

                filtered_ancest = F.conv1d(spike_ancest, ancest_kern, groups=self.sub_no)
                filtered_hist = F.conv1d(spike_hist, hist_kern, groups=self.sub_no-1)
                filtered_ancest = filtered_ancest.flatten()
                filtered_hist = filtered_hist.flatten()
        
                X_in = S_conv[t] + filtered_ancest + raw_ancest + self.theta_syn
                X_in[1:] = X_in[1:] + filtered_hist
                X_out = torch.sigmoid(X_in)
                X[t] = X_out
                Y[t+1] = X_out * self.W_sub

                Z_ratios_prior = torch.sigmoid(X_out[1:] * self.W_spike + self.theta_spike)
                Z_ratios_post = posterior_probs_ratios[t]
                Z_ratios_final = 0.5*Z_ratios_prior + 0.5*Z_ratios_post
                #Z_ratios_final = posterior_probs_ratios[t]

                uniform = torch.rand(self.sub_no-1).cuda()
                logistic = torch.log(uniform +1e-10) - torch.log(1-uniform +1e-10)
                Z_pre = (torch.log(Z_ratios_final +1e-10) + logistic) / self.temp
                Z_post = torch.sigmoid(Z_pre)

                Z_post_ratios_out[t] = Z_post_ratios_out[t] + Z_ratios_final
                Z_prior_ratios_out[t] = Z_prior_ratios_out[t] + Z_ratios_prior
                Z_samples[t+self.T_hist] = Z_samples[t+self.T_hist] + Z_post
                spike_hist = torch.cat((spike_hist.reshape(-1,self.T_hist)[:,1:], Z_post.reshape(-1,1)),1)

            final_voltage = Y[1:,0] + self.V_o
            final_Y = Y[1:,1:]
            final_Z_samples = Z_samples[self.T_hist:,:]

            return final_voltage, final_Y, final_Z_samples, Z_prior_ratios_out, Z_post_ratios_out

    def forward(self, V_in, S_e, S_i, beta):

        posterior_probs_ratios = self.encode(V_in, S_e, S_i)
        V_out, Y_out, Z_spikes, Z_prior_ratios, Z_post_ratios = self.decode(S_e, S_i, posterior_probs_ratios)
        
        ### Reconstruction Loss ###
        #rec_loss = torch.mean((V_out - V_in)**2)
        rec_loss = torch.var(V_out - V_in)

        Z_P_prior = Z_prior_ratios / (1+Z_prior_ratios)
        Z_P_post = Z_post_ratios / (1+Z_post_ratios)

        KL_loss_full = Z_P_post * torch.log(Z_P_post/Z_P_prior +1e-10)  + (1-Z_P_post) * torch.log((1-Z_P_post)/(1-Z_P_prior) +1e-10)
        KL_loss = torch.mean(torch.sum(KL_loss_full, 1) , 0)

        return V_out, rec_loss, KL_loss, Z_P_prior, Z_P_post
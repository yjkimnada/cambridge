import torch
import torch.nn as nn
import torch.nn.functional as F

class step_shGLM(nn.Module):
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
        self.W_s_syn = nn.Parameter(torch.randn(self.sub_no, 2) * 0.02, requires_grad=True)
        self.W_ns_syn =  nn.Parameter(torch.randn(self.sub_no, self.syn_basis_no, 2) * 0.02, requires_grad=True)
        self.Tau_ns_syn = nn.Parameter(torch.arange(0,self.syn_basis_no*1,1).float().reshape(-1,1).repeat(1,2) , requires_grad=False)
        self.Delta_ns_syn = nn.Parameter(torch.rand(self.sub_no, 2), requires_grad=True)

        ### Ancestor Subunit Parameters ###
        self.W_ns_sub = nn.Parameter(torch.rand(self.sub_no) , requires_grad=True)

        ### Subunit Output Parameters ###
        self.V_o = nn.Parameter(torch.randn(1), requires_grad=True)
        self.Theta_s = nn.Parameter(torch.rand(self.sub_no-1) * -0.2, requires_grad=True)
        self.Theta_ns = nn.Parameter(torch.rand(self.sub_no) * -0.2, requires_grad=True)

        ### History Parameters ###
        self.W_ns_hist = nn.Parameter(torch.rand(self.sub_no - 1, self.hist_basis_no) * (-0.02), requires_grad=True)
        self.Tau_ns_hist = nn.Parameter(torch.arange(0,self.hist_basis_no*1,1).float() , requires_grad=False)
        self.Delta_ns_hist = nn.Parameter(torch.rand(self.sub_no), requires_grad=True)

        ### Propagation Parameters ###
        self.W_s_prop = nn.Parameter(torch.rand(self.sub_no-1) * 0.02, requires_grad=True)
        self.W_ns_prop = nn.Parameter(torch.rand(self.sub_no , self.prop_basis_no) * 0.02, requires_grad=True)
        self.Tau_ns_prop = nn.Parameter(torch.arange(0,self.prop_basis_no*1,1).float() , requires_grad=False)
        self.Delta_ns_prop = nn.Parameter(torch.rand(self.sub_no), requires_grad=True)

        self.spike_derivScale = torch.ones(self.sub_no - 1).cuda() * 0.5
        self.spike_derivTime = torch.ones(self.sub_no - 1).cuda() * 1
        self.spike_decay = torch.rand(self.sub_no - 1).cuda() * 0.4
        

    def forward(self, S_e, S_i):
        T_data = S_e.shape[0]

        ############
        # Pre-convolve synapse inputs
        #############
        syn_s_in = torch.zeros(T_data, self.sub_no).cuda()
        syn_ns_in = torch.zeros(T_data, self.sub_no).cuda()
        
        for s in range(self.sub_no):
            t_raw = torch.arange(self.T_syn).cuda()

            delta_ns_e = torch.exp(self.Delta_ns_syn[s,0])
            delta_ns_i = torch.exp(self.Delta_ns_syn[s,1])

            t_ns_e = t_raw - delta_ns_e
            t_ns_i = t_raw - delta_ns_i
            t_ns_e[t_ns_e < 0.0] = 0.0
            t_ns_i[t_ns_i < 0.0] = 0.0
            full_ns_e_kern = torch.zeros(self.T_syn).cuda()
            full_ns_i_kern = torch.zeros(self.T_syn).cuda()

            for b in range(self.syn_basis_no):
                tau_ns_e = torch.exp(self.Tau_ns_syn[b,0]) 
                tau_ns_i = torch.exp(self.Tau_ns_syn[b,1])
                t_ns_e_tau = t_ns_e / tau_ns_e
                t_ns_i_tau = t_ns_i / tau_ns_i

                part_ns_e_kern = t_ns_e_tau * torch.exp(-t_ns_e_tau)
                part_ns_i_kern = t_ns_i_tau * torch.exp(-t_ns_i_tau)
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

            full_ns_e_kern = torch.flip(full_ns_e_kern, [0])
            full_ns_i_kern = torch.flip(full_ns_i_kern, [0])
            full_ns_e_kern = full_ns_e_kern.reshape(1,1,-1)
            full_ns_i_kern = full_ns_i_kern.reshape(1,1,-1)

            filtered_ns_e = F.conv1d(pad_in_e, full_ns_e_kern, padding=0).squeeze(1).T
            filtered_ns_i = F.conv1d(pad_in_i, full_ns_i_kern, padding=0).squeeze(1).T

            syn_ns_in[:,s] = syn_ns_in[:,s] + filtered_ns_e.flatten() + filtered_ns_i.flatten()
            syn_s_in[:,s] = in_e * self.W_s_syn[s,0] + in_i * self.W_s_syn[s,1]
        
        #############
        # Solve for X_t, Y_t, Z_t through time
        ###########
        
        Z_pad = torch.zeros(T_data + self.T_hist, self.sub_no - 1).cuda() ### spike trains
        X_s_pad = torch.zeros(T_data+1, self.sub_no - 1).cuda() ### sub-threshold values for spiking subunit
        Y_ns_pad = torch.zeros(T_data+1, self.sub_no).cuda() ### scaled ancestor subunit inputs to non-spiking subunit

        hist_ns_kern = torch.zeros(self.sub_no - 1, self.T_hist).cuda()
        for s in range(self.sub_no - 1):
            t = torch.arange(self.T_hist).cuda()
            for b in range(self.hist_basis_no):
                tau_ns = torch.exp(self.Tau_ns_hist[b])
                t_ns_tau = t / tau_ns
                hist_ns_kern[s,:] = hist_ns_kern[s,:] + t_ns_tau * torch.exp(-t_ns_tau) * self.W_ns_hist[s,b]
        hist_ns_kern = torch.flip(hist_ns_kern, [1]).reshape(self.sub_no - 1, 1, -1)

        prop_ns_kern = torch.zeros(self.sub_no, self.T_hist).cuda()
        for s in range(self.sub_no):
            t = torch.arange(self.T_hist).cuda()
            for b in range(self.prop_basis_no):
                tau_ns = torch.exp(self.Tau_ns_prop[b])
                t_ns_tau = t / tau_ns
                prop_ns_kern[s,:] = prop_ns_kern[s,:] + t_ns_tau * torch.exp(-t_ns_tau) * self.W_ns_prop[s,b]
        prop_ns_kern = torch.flip(prop_ns_kern, [1]).reshape(self.sub_no, 1, -1)

        remove_first_sub = torch.eye(self.sub_no)[1:,:].cuda()
        
        for t in range(T_data):
            spike_ns_hist = Z_pad[t:self.T_hist+t , :].T.clone() # sub_no - 1
            spike_s_last = Z_pad[t+self.T_hist-1 , :].clone() # sub_no - 1
            raw_ns_prop = torch.matmul(self.C_den, Y_ns_pad[t]) # sub_no
            spike_ns_prop = torch.matmul(self.C_den[:,1:], spike_ns_hist) # sub_no
            #spike_s_prop = torch.matmul(self.C_den[1:,1:], spike_s_last)
            spike_ns_hist = spike_ns_hist.reshape(1, self.sub_no - 1, -1)
            spike_ns_prop = spike_ns_prop.reshape(1, self.sub_no, -1)

            filtered_ns_prop = F.conv1d(spike_ns_prop, prop_ns_kern, groups=self.sub_no).flatten() # sub_no
            filtered_ns_hist = F.conv1d(spike_ns_hist, hist_ns_kern, groups=self.sub_no-1).flatten() # sub_no - 1
            X_ns_in = syn_ns_in[t] + filtered_ns_prop + raw_ns_prop + self.Theta_ns # sub_no
            X_ns_in[1:] = X_ns_in[1:] + filtered_ns_hist
            X_ns_out = torch.sigmoid(X_ns_in)
            Y_ns_pad[t+1] = X_ns_out * self.W_ns_sub

            X_s_in = X_s_pad[t]*self.spike_decay + syn_s_in[t,1:] + torch.matmul(self.C_den[1:,1:],spike_s_last*self.W_s_prop) + self.Theta_s
            Z_out = Spike_Function.apply(X_s_in, self.spike_derivScale, self.spike_derivTime)
            Z_pad[t+self.T_hist] = Z_out

            X_s_pad[t+1] = X_s_in
            X_s_pad[t+1][Z_out == 1] = 0

        final_voltage = Y_ns_pad[1:,0] + self.V_o
        final_Y = Y_ns_pad[1:,1:]
        final_X = X_s_pad[1:,:]
        final_Z = Z_pad[self.T_hist:,:]
        
        #print(spike_s_last)
        #print(spike_s_prop)
        #print(X_s_pad[t])
        #print(syn_s_in[t])
        #print(spike_s_prop)
        #print(self.Theta_s)
        
        return final_voltage, final_Y, final_Z, final_X

            
class Spike_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_value, derivScale, derivTime):
        #device = torch.cuda.current_device()
        #print(input_value.shape)
        
        output_value = torch.empty(input_value.shape[0]).cuda()

        for x in range(input_value.shape[0]):
            if input_value[x] < 0:
                output_value[x] = 0
            else:
                output_value[x] = 1

        derivScale = torch.autograd.Variable(derivScale, requires_grad=False)
        derivTime = torch.autograd.Variable(derivTime, requires_grad=False)
        ctx.save_for_backward(input_value, derivScale, derivTime)

        return output_value
    
    def backward(ctx, gradOutput):

        (input_value, derivScale, derivTime) = ctx.saved_tensors
        spikeDerivative = derivScale * torch.exp( -torch.abs(input_value) / derivTime)

        return gradOutput * spikeDerivative, None,  None, None
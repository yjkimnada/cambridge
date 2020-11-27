import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)

        self.left_padding = dilation * (kernel_size - 1)

    def forward(self, input):
        # input is size (samples=1, chan, time)
        x = F.pad(input, (self.left_padding , 0))

        return super(CausalConv1d, self).forward(x)


def causalconv1d(x, weight, stride=1, dilation=1, groups=1):
    padding = dilation * (weight.shape[-1] - 1)
    pad_input = F.pad(x, (padding , 0))
    out = F.conv1d(pad_input, weight, stride=stride, dilation=dilation, groups=groups)
    return out


class hist_hTCN(nn.Module):
    def __init__(self, C_den, C_syn_e, C_syn_i, T_no, M_no, B_no):
        super().__init__()

        self.sub_no = C_den.shape[0]
        self.C_den = C_den
        self.C_syn_e = C_syn_e
        self.C_syn_i = C_syn_i
        self.E_no = C_syn_e.shape[1]
        self.I_no = C_syn_i.shape[1]
        self.T_no = T_no
        self.M_no = M_no
        self.B_no = B_no
        
        self.conv1_e_bases = nn.Parameter(torch.randn(self.B_no, self.T_no) , requires_grad=True)
        self.conv1_e_weights = nn.Parameter(torch.randn(self.M_no*self.sub_no , self.B_no) , requires_grad=True)
        self.conv1_i_bases = nn.Parameter(torch.randn(self.B_no, self.T_no) , requires_grad=True)
        self.conv1_i_weights = nn.Parameter(torch.randn(self.M_no*self.sub_no , self.B_no) , requires_grad=True)
        
        self.conv_hist_bases = nn.Parameter(torch.randn(self.B_no, self.T_no) , requires_grad=True)
        self.conv_hist_weights = nn.Parameter(torch.randn(self.sub_no* self.M_no , self.B_no) , requires_grad=True)
        
        
        self.leaf_linear = nn.Parameter(torch.randn(self.sub_no, self.M_no) , requires_grad=True)
        self.multiplex_linear = nn.Parameter(torch.randn(self.sub_no, self.M_no) , requires_grad=True)
        self.multiplex_bias = nn.Parameter(torch.randn(self.sub_no) , requires_grad=True)

        self.nonlin = nn.Tanh()
        
    def forward(self, S_e, S_i):
        T_data = S_e.shape[0]
        S_e = torch.matmul(S_e, self.C_syn_e.T).T.reshape(1, self.sub_no, -1)
        S_i = torch.matmul(S_i, self.C_syn_i.T).T.reshape(1, self.sub_no, -1)

        #### Initial spike convolutions ####
        
        conv1_e_kern = torch.matmul(self.conv1_e_weights, self.conv1_e_bases).unsqueeze(1)
        conv1_i_kern = torch.matmul(self.conv1_i_weights, self.conv1_i_bases).unsqueeze(1)
        S_e_conv1 = causalconv1d(S_e, conv1_e_kern, groups = self.sub_no)
        S_i_conv1 = causalconv1d(S_i, conv1_i_kern, groups = self.sub_no)
        
        S_e_conv1 = S_e_conv1[0].T.reshape(T_data, self.sub_no, self.M_no)
        S_i_conv1 = S_i_conv1[0].T.reshape(T_data, self.sub_no, self.M_no)
        
        S_conv = S_e_conv1 + S_i_conv1 # (T_data, sub_no, M_no)
        
        #### Combine convolved spike trains ####
        sub_out = torch.zeros(T_data + self.T_no, self.sub_no).cuda() # pre scalar post nonlin
        
        hist_kern = torch.matmul(self.conv_hist_weights, self.conv_hist_bases).unsqueeze(1) # (sub_no* M_no,1 , T_no)
        
        for t in range(T_data):
            ### History
            sub_hist_out = sub_out[t:t+self.T_no,:].clone() # (T_no, sub_no)
            sub_hist_out = sub_hist_out.T.unsqueeze(0) # (1, sub_no, T_no)
            sub_hist_in = F.conv1d(sub_hist_out, hist_kern, groups=self.sub_no).reshape(self.sub_no, self.M_no)######
            
            ### Leaf
            leaf_in = sub_out[t+self.T_no - 1,:].clone() #(sub_no)
            leaf_scale = leaf_in * self.leaf_linear.T # (M_no , sub_no)
            leaf_in = leaf_scale.T
            
            ### Syn
            syn_in = S_conv[t,:,:] # (sub_no, M_no)
            
            ### Combine
            nonlin_out = self.nonlin(syn_in + leaf_in + sub_hist_in) # (sub_no , M_no)
            plex_out = nonlin_out * self.multiplex_linear 
            plex_out = torch.sum(plex_out , 1) + self.multiplex_bias
            sub_out[t+self.T_no] = sub_out[t+self.T_no] + plex_out

        return sub_out[self.T_no:,0]


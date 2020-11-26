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


class ff2_hTCN(nn.Module):
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
        
        
        self.leaf_linear = nn.Parameter(torch.randn(self.sub_no, self.M_no) , requires_grad=True)
        self.multiplex_linear = nn.Parameter(torch.randn(self.sub_no, self.M_no) , requires_grad=True)
        self.multiplex_bias = nn.Parameter(torch.randn(self.sub_no) , requires_grad=True)
        
        self.ff1_linear = nn.Parameter(torch.randn(self.sub_no, self.M_no, self.M_no) , requires_grad=True)
        self.ff2_linear = nn.Parameter(torch.randn(self.sub_no, self.M_no, self.M_no) , requires_grad=True)

        #self.nonlin = nn.Tanh()
        self.nonlin = nn.PReLU()
        
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
        sub_out = torch.zeros(T_data, self.sub_no).cuda()
        
        for s in range(self.sub_no):
            sub_idx = -s-1
            leaf_idx = torch.where(self.C_den[sub_idx] == 1)[0]

            if torch.numel(leaf_idx) == 0:
                nonlin_out = self.nonlin(S_conv[:,sub_idx,:]) # (T_data, M_no) 
                
                #####
                nonlin_out = torch.matmul(nonlin_out , self.ff1_linear[sub_idx])
                nonlin_out = self.nonlin(nonlin_out)
                nonlin_out = torch.matmul(nonlin_out, self.ff2_linear[sub_idx])
                #nonlin_out = self.nonlin(nonlin_out)
                nonlin_out = torch.tanh(nonlin_out)
                #######
                
                sub_out[:,sub_idx] = sub_out[:,sub_idx] + torch.matmul(nonlin_out, torch.exp(self.multiplex_linear[sub_idx,:])) + self.multiplex_bias[sub_idx]
            else:
                syn_in = S_conv[:,sub_idx,:] # (T_data, M_no)
                leaf_in = torch.matmul(sub_out[:,leaf_idx] , torch.exp(self.leaf_linear[leaf_idx,:])) # (T_data, M_no)
                nonlin_in = syn_in + leaf_in # (T_data, M_no)
                nonlin_out = self.nonlin(nonlin_in)
                
                ######
                nonlin_out = torch.matmul(nonlin_out , self.ff1_linear[sub_idx])
                nonlin_out = self.nonlin(nonlin_out)
                nonlin_out = torch.matmul(nonlin_out, self.ff2_linear[sub_idx])
                #nonlin_out = self.nonlin(nonlin_out)
                nonlin_out = torch.tanh(nonlin_out)
                #######
                
                sub_out[:,sub_idx] = sub_out[:,sub_idx] + torch.matmul(nonlin_out, torch.exp(self.multiplex_linear[sub_idx,:])) + self.multiplex_bias[sub_idx]

        return sub_out[:,0]


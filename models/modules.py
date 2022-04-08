import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import copy
from collections import OrderedDict


class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_p=0.25):
        super(LinearBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ELU(),
            nn.Dropout(dropout_p),
        )
    def forward(self, input_):
        
        out = self.layer(input_)
        return out
    
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, bias=True, w_init='linear'):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation,
                              bias=bias)

        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        x = self.conv(x)
        return x
class Mel_Encoder_Prenet(nn.Module):
    """
    Prenet before passing through the network
    """
    def __init__(self, input_size, hidden_size, output_size, dropout_p=0.5):
        """
        :param input_size: dimension of input
        :param hidden_size: dimension of hidden unit
        :param output_size: dimension of output
        """
        super(Mel_Encoder_Prenet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        
        self.layer = nn.Sequential(OrderedDict([
             ('fc1', nn.Linear(self.input_size, self.hidden_size)),
             ('relu1', nn.ReLU()),
             ('dropout1', nn.Dropout(dropout_p)),
             ('fc2', nn.Linear(self.hidden_size, self.hidden_size)),
             ('relu2', nn.ReLU()),
             ('dropout2', nn.Dropout(dropout_p)),
             ('fc3', nn.Linear(self.hidden_size, self.output_size)),
             ('relu3', nn.ReLU()),
        ]))

    def forward(self, input_):

        out = self.layer(input_)

        return out
    
class FFN(nn.Module):
    """
    Feed Forward Network
    """
    def __init__(self, hidden_size):
        super(FFN, self).__init__()
        self.w_1 = Conv(hidden_size, hidden_size * 4, 
                        kernel_size=9, w_init='relu', padding='same')
        self.w_2 = Conv(hidden_size * 4, hidden_size, 
                        kernel_size=9, padding='same')
        
    def forward(self, input_):
        x = input_.transpose(1, 2)
        x = self.w_2(torch.relu(self.w_1(x)))
        x = x.transpose(1, 2)
        
        return x
    
class EncoderBlock(nn.Module):
    """
    Encoder Block
    """

    def __init__(self, hidden_size, n_split, dropout_p):
        """
        Multihead Attention(MHA) : Q, K and V are equal
        """
        super(EncoderBlock, self).__init__()
        self.MHA = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=n_split, batch_first=True)
        self.MHA_dropout = nn.Dropout(dropout_p)
        self.MHA_norm = nn.LayerNorm(hidden_size)
        self.FFN = FFN(hidden_size)
        self.FFN_norm = nn.LayerNorm(hidden_size)
        self.FFN_dropout = nn.Dropout(dropout_p)

    def forward(self, input_, mask):
        # [Xiong et al., 2020] shows that pre-layer normalization works better
        x = self.MHA_norm(input_)
        x, attn = self.MHA(query=x, key=x, value=x, key_padding_mask=mask)
        x = input_ + self.MHA_dropout(x)
        x = x + self.FFN_dropout(self.FFN(self.FFN_norm(x)))
        return x, attn

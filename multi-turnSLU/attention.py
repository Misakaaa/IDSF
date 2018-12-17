
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import math

class Attn(nn.Module):
    def __init__(self, method, hidden_size, T):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.T = T
        self.attn = nn.Linear(hidden_size*4, hidden_size*2)
        self.v = nn.Parameter(torch.rand(hidden_size*2))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs):
        '''
        :param hidden:
            previous hidden state of the decoder, in shape (layers*directions,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (T,B,H)
        :return
            attention energies in shape (B,T)
        '''
        max_len = encoder_outputs.size(0)
        # print(max_len)
        this_batch_size = encoder_outputs.size(1)
        # H = hidden.repeat(max_len,1,1).transpose(0,1)
        H = hidden.transpose(0,1)
        encoder_outputs = encoder_outputs.transpose(0,1) # [B*T*H]
        attn_energies = self.score(H,encoder_outputs) # compute attention score
        return F.softmax(attn_energies.squeeze(-1), dim=1) # normalize with softmax

    def score(self, hidden, encoder_outputs):
        a = torch.cat([hidden, encoder_outputs], 2)
        energy = F.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2))) # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2,1) # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0],1).unsqueeze(1) #[B*1*H]
        energy = torch.bmm(v,energy) # [B*1*T]
        return energy.squeeze(1) #[B*T]
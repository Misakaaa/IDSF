import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from self_attention import SelfAttention
from self_att import SelfAttention as SelfA
from attention import Attn
import numpy as np


class SDEN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, slot_size, intent_size, dropout=0.3, pad_idx=0):
        super(SDEN, self).__init__()

        self.pad_idx = 0
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=self.pad_idx)
        self.bigru_m = nn.GRU(embed_size, hidden_size, batch_first=True, bidirectional=True)
        self.bigru_c = nn.GRU(embed_size, hidden_size, batch_first=True, bidirectional=True)
        self.context_encoder = nn.Sequential(nn.Linear(hidden_size * 4, hidden_size * 2),
                                             nn.Sigmoid())
        self.Att = Attn('concat', hidden_size)
        self.context_encoder1 = nn.Sequential(nn.Linear(hidden_size * 8, hidden_size * 2),
                                              nn.Sigmoid())

        self.session_encoder = nn.GRU(hidden_size * 2, hidden_size * 2, batch_first=True, bidirectional=True)

        self.decoder_1 = nn.GRU(embed_size, hidden_size * 2, batch_first=True, bidirectional=True)
        self.decoder_2 = nn.LSTM(hidden_size * 4, hidden_size * 2, batch_first=True, bidirectional=True)

        self.intent_linear = nn.Linear(hidden_size * 4, intent_size)
        self.slot_linear = nn.Linear(hidden_size * 4, slot_size)
        self.dropout = nn.Dropout(dropout)
        self.attention = SelfAttention(hidden_size)
        self.att = SelfA(hidden_size)
        self.hidden_size = hidden_size
        # self.att = Attn('concat', 64)

        for param in self.parameters():
            if len(param.size()) > 1:
                nn.init.xavier_uniform_(param)
            else:
                param.data.zero_()

    def forward(self, history, current):
        batch_size = len(history)
        H = []  # encoded history
        for h in history:  # 编码对于当前c来说的整个历史信息
            mask = (h != self.pad_idx)
            length = mask.sum(1).long()
            embeds = self.embed(h)
            embeds = self.dropout(embeds)
            lens, indices = torch.sort(length, 0, True)
            lens = [l if l > 0 else 1 for l in lens.tolist()]  # all zero-input
            packed_h = pack(embeds[indices], lens, batch_first=True)
            outputs, hidden = self.bigru_m(packed_h)
            _, _indices = torch.sort(indices, 0)
            hidden = torch.cat([hh for hh in hidden], -1)
            hidden = hidden[_indices].unsqueeze(0)
            H.append(hidden)

        M = torch.cat(H)  # B,T_C,2H
        M = self.dropout(M)  # 上下文历史信息

        embeds = self.embed(current)  # 当前句的embedding
        embeds = self.dropout(embeds)
        mask = (current != self.pad_idx)
        length = mask.sum(1).long()
        lens, indices = torch.sort(length, 0, True)
        packed_h = pack(embeds[indices], lens.tolist(), batch_first=True)
        outputs, hidden = self.bigru_c(packed_h)  # 过了一层双向GRU后的当前句c
        _, _indices = torch.sort(indices, 0)
        hidden = torch.cat([hh for hh in hidden], -1)  # 将c的隐层都拼起来
        C = hidden[_indices].unsqueeze(1)  # B,1,2H
        C = self.dropout(C)

        C = C.repeat(1, M.size(1), 1)
        # C1 = torch.zeros(C.size())
        CONCAT = torch.cat([M, C], -1)  # B,T_c,4H

        G = self.context_encoder(CONCAT)  # 过一道前馈神经网络

        # attention 对G做attention
        #Att = Attn('concat', self.hidden_size).cuda()
        weights = self.Att(C, G)
        outputs = G * weights.unsqueeze(-1)

        # decoder
        _, H = self.session_encoder(outputs)  # 2,B,2H 将所有的历史信息压缩成 H
        # H = torch.zeros(H1.size())
        weight = next(self.parameters())
        cell_state = weight.new_zeros(H.size())
        O_1, _ = self.decoder_1(embeds)  # 0_1 是第一层decoder的RNN最后一层的输出
        O_1 = self.dropout(O_1)

        O_2, (S_2, _) = self.decoder_2(O_1, (H, cell_state))  # 0_2 是第二层decoder的RNN最后一层的输出，S_2 是最后一个时间步的隐层状态，用于意图识别
        O_2 = self.dropout(O_2)
        S = torch.cat([s for s in S_2], 1)

        intent_att, _ = self.attention(O_2)  # 对0_2 做 自注意力机制
        H = torch.cat([H[0], H[1]], 1)
        H = torch.unsqueeze(H, dim=1)

        H = H.repeat(1, intent_att.size(1), 1)

        CON = torch.cat([H, intent_att], -1)
        g1 = self.context_encoder1(CON)
        g = g1.sum(2)

        g = torch.unsqueeze(g, dim=2)
        g = g.repeat(1, 1, self.hidden_size*4)

        hg = intent_att.mul(g)
        hg = hg.add(O_2)

        intent_prob = self.intent_linear(S)

        slot_prob = self.slot_linear(hg.contiguous().view(hg.size(0) * hg.size(1), -1))

        return slot_prob, intent_prob



if __name__ == '__main__':
    ts = torch.Tensor(16, 3, 64, 64)
    # vr = Variable(ts)
    net = SDEN(vocab_size=1179 , embed_size=100, hidden_size=64, slot_size=24, intent_size=4)
    print(net)

    # oups = net(vr)
    # for oup in oups:
    #     print(oup.size())

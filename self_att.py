import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim*2, 64),
            # nn.Linear(hidden_dim * 4, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        # (B, L, H) -> (B , L, 1)
        energy = self.projection(encoder_outputs)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        # (B, L, H) * (B, L, 1) -> (B, H)
        outputs = (encoder_outputs * weights.unsqueeze(-1))
        return outputs, weights

class AttnClassifier(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.attention = SelfAttention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def set_embedding(self, vectors):
        self.embedding.weight.data.copy_(vectors)

    def forward(self, inputs, lengths):
        batch_size = inputs.size(1)
        # (L, B)
        embedded = self.embedding(inputs)
        # (L, B, E)
        packed_emb = nn.utils.rnn.pack_padded_sequence(embedded, lengths)
        out, hidden = self.lstm(packed_emb)
        out = nn.utils.rnn.pad_packed_sequence(out)[0]
        out = out[:, :, :self.hidden_dim] + out[:, :, self.hidden_dim:]
        # (L, B, H)
        embedding, attn_weights = self.attention(out.transpose(0, 1))
        # (B, HOP, H)
        outputs = self.fc(embedding.view(batch_size, -1))
        # (B, 1)
        return outputs, attn_weights

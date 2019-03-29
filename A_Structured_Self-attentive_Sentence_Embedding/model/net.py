import torch
import torch.nn as nn
import torch.nn.functional as F
from gluonnlp import Vocab
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SelfAttention(nn.Module):
    """SelfAttention class"""
    def __init__(self, input_dim: int, da: int, r: int) -> None:
        """Instantiating SelfAttention class

        Args:
            input_dim (int): dimension of input, eg) (batch_size, seq_len, input_dim)
            da (int): dimension of hidden layer
            r (int): number of aspects
        """
        super(SelfAttention, self).__init__()
        self._ws1 = nn.Linear(input_dim, da, bias=False)
        self._ws2 = nn.Linear(da, r, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        attn_mat = F.softmax(self._ws2(F.tanh(self._ws1(h))), dim=1)
        attn_mat = attn_mat.permute(0, 2, 1)
        return attn_mat

class SelfAttentiveNet(nn.Module):
    """SelfAttentiveNet class"""
    def __init__(self, num_classes: int, lstm_hidden_dim: int, da: int, r: int,
                 hidden_dim: int, vocab: Vocab) -> None:
        """Instantiating SelfAttentiveNet class

        Args:
            num_classes (int): number of classes
            lstm_hidden_dim (int): hidden dimension of bi-directional lstm
            da (int): hidden dimension of self-attention
            r (int): number of aspect of self-attention
            hidden_dim (int): hidden dimension of mlp
            vocab (gluonnlp.Vocab): instance of gluonnlp.Vocab
        """
        super(SelfAttentiveNet, self).__init__()
        self._embedding = nn.Embedding.from_pretrained(torch.from_numpy(vocab.embedding.idx_to_vec.asnumpy()))
        self._lstm = nn.LSTM(self._embedding.embedding_dim, hidden_size=lstm_hidden_dim, bidirectional=True,
                             batch_first=True)
        self._attention = SelfAttention(2 * lstm_hidden_dim, da, r)
        self._fc1 = nn.Linear(2 * lstm_hidden_dim * r, hidden_dim)
        self._fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, x_len: torch.Tensor) -> None:
        x_batch = self._embedding(x)
        x_batch = pack_padded_sequence(x_batch, x_len, True)
        outputs, _ = self._lstm(x_batch)
        hiddens = pad_packed_sequence(outputs)[0].permute(1, 0, 2)
        attn_mat = self._attention(hiddens)
        m = torch.bmm(attn_mat, hiddens)
        m = m.view(m.size(0), -1)
        hidden = F.relu(self._fc1(m))
        score = self._fc2(hidden)
        return score, attn_mat
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.ops import PreEmbedding, Linker, BiLSTM, SelfAttention
from gluonnlp import Vocab
from typing import Tuple


class SAN(nn.Module):
    """SAN class"""
    def __init__(self, num_classes: int, lstm_hidden_dim: int, da: int, r: int, hidden_dim: int, vocab: Vocab) -> None:
        """Instantiating SAN class

        Args:
            num_classes (int): the number of classes
            lstm_hidden_dim (int): the number of features in the hidden states in bi-directional lstm
            da (int): the number of features in hidden layer from self-attention
            r (int): the number of aspects of self-attention
            hidden_dim (int): the number of features in hidden layer from mlp
            vocab (gluonnlp.Vocab): the instance of gluonnlp.Vocab
        """
        super(SAN, self).__init__()
        self._embedding = PreEmbedding(vocab, padding_idx=1, freeze=False, permuting=False, tracking=True)
        self._pipe = Linker(permuting=False)
        self._bilstm = BiLSTM(self._embedding._ops.embedding_dim, lstm_hidden_dim, using_sequence=True)
        self._attention = SelfAttention(2 * lstm_hidden_dim, da, r)
        self._fc1 = nn.Linear(2 * lstm_hidden_dim * r, hidden_dim)
        self._fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.tensor, torch.tensor]:
        fmap = self._embedding(x)
        fmap = self._pipe(fmap)
        hiddens = self._bilstm(fmap)
        attn_mat = self._attention(hiddens)
        m = torch.bmm(attn_mat, hiddens)
        m = m.view(m.size()[0], -1)
        hidden = F.relu(self._fc1(m))
        score = self._fc2(hidden)
        return score, attn_mat
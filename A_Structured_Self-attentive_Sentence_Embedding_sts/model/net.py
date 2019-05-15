import torch
import torch.nn as nn
import torch.nn.functional as F
from model.ops import SentenceEncoder
from gluonnlp import Vocab
from typing import Tuple


class SAN(nn.Module):
    """SAN class"""

    def __init__(self, num_classes: int, lstm_hidden_dim: int, hidden_dim: int, da: int, r: int, vocab: Vocab) -> None:
        """

        Args:
            num_classes (int): the number of classes
            lstm_hidden_dim (int): the number of features in the hidden states in bi-directional lstm
            hidden_dim (int): the number of features in hidden layer from mlp
            da (int): the number of features in hidden layer from self-attention
            r (int): the number of aspects of self-attention
            vocab (gluonnlp.Vocab): the instance of gluonnlp.Vocab
        """
        super(SAN, self).__init__()
        self._encoder = SentenceEncoder(lstm_hidden_dim, da, r, 1, vocab)
        self._wa = nn.Parameter(torch.randn(lstm_hidden_dim * 2, lstm_hidden_dim * 2))
        self._wb = nn.Parameter(torch.randn(lstm_hidden_dim * 2, lstm_hidden_dim * 2))
        self._fc1 = nn.Linear(r * lstm_hidden_dim * 2, hidden_dim)
        self._fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_a, query_b = x
        query_a_emb, query_a_attn_mat = self._encoder(query_a)
        query_b_emb, query_b_attn_mat = self._encoder(query_b)
        fa = query_b_emb @ self._wa
        fb = query_b_emb @ self._wb
        fab = fa * fb
        fab = fab.view(fab.size()[0], -1)
        feature = F.relu(self._fc1(fab))
        score = self._fc2(feature)

        return score, query_a_attn_mat, query_b_attn_mat

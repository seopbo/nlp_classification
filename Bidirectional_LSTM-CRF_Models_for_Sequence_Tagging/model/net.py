import torch
import torch.nn as nn
from model.ops import PreEmbedding, Linker, BiLSTM
from gluonnlp import Vocab
from typing import Tuple


class BilstmCRF(nn.Module):
    """BilstmCRF"""
    def __init__(self, label_vocab: Vocab, token_vocab: Vocab, lstm_hidden_dim: int) -> None:
        """Instantiating BilstmCRF class

        Args:
            token_vocab: (gluonnlp.Vocab): the instance of gluonnlp.Vocab that has token information
            label_vocab: (gluonnlp.Vocab): the instance of gluonnlp.Vocab that has label information
            lstm_hidden_dim (int): the number of hidden dimension of lstm
        """
        super(BilstmCRF, self).__init__()
        self._embedding = PreEmbedding(token_vocab, padding_idx=1, freeze=False, permuting=False, tracking=True)
        self._pipe = Linker(permuting=False)
        self._bilstm = BiLSTM(self._embedding._ops.embedding_dim, lstm_hidden_dim, using_sequence=True)
        self._fc = nn.Linear(2 * lstm_hidden_dim, len(label_vocab))

    def forward(self, x: torch.Tensor) -> Tuple[torch.tensor, torch.tensor]:
        fmap = self._embedding(x)
        fmap = self._pipe(fmap)
        hiddens = self._bilstm(fmap)
        scores = self._fc(hiddens)
        return scores
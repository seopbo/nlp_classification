import torch
import torch.nn as nn
from model.ops import PreEmbedding, Linker, LSTMEncoder
from model.utils import Vocab
from typing import Tuple


class MaLSTM(nn.Module):
    """MaLSTM class"""

    def __init__(self, num_classes: int, hidden_dim: int, vocab: Vocab) -> None:
        """Instantiating MaLSTM class

        Args:
            num_classes (int): the number of classes
            hidden_dim (int): the number of features of the hidden states from LSTM
            vocab (model.utils.Vocab): the instance of model.utils.Vocab
        """
        super(MaLSTM, self).__init__()
        self._emb = PreEmbedding(
            vocab, padding_idx=1, freeze=False, permuting=False, tracking=True
        )
        self._pipe = Linker(permuting=False)
        self._encoder = LSTMEncoder(
            self._emb._ops.embedding_dim, hidden_dim, using_sequence=False
        )
        self._classifier = nn.Linear(in_features=hidden_dim, out_features=num_classes)

    def forward(self, x: Tuple[torch.tensor, torch.tensor]) -> torch.Tensor:
        qa, qb = x
        fmap_qa = self._encoder(self._pipe(self._emb(qa)))
        fmap_qb = self._encoder(self._pipe(self._emb(qb)))
        fmap = torch.exp(-torch.abs(fmap_qa - fmap_qb))
        score = self._classifier(fmap)
        return score

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.ops import Embedding, Conv1d, MaxPool1d, Linker, BiLSTM
from gluonnlp import Vocab


class ConvRec(nn.Module):
    """ConvRec class"""
    def __init__(self, num_classes: int, embedding_dim: int, hidden_dim: int, vocab: Vocab) -> None:
        """Instantiating ConvRec class

        Args:
            num_classes (int): the number of classes
            embedding_dim (int) : the dimension of embedding vector for token
            vocab (gluonnlp.Vocab): the instance of gluonnlp.Vocab
        """
        super(ConvRec, self).__init__()
        self._ops = nn.Sequential(Embedding(len(vocab), embedding_dim, vocab.to_indices(vocab.padding_token),
                                            permuting=True, tracking=True),
                                  Conv1d(embedding_dim, hidden_dim, 5, 1, 1, F.relu, tracking=True),
                                  MaxPool1d(2, 2, tracking=True),
                                  Conv1d(hidden_dim, hidden_dim, 3, 1, 1, F.relu, tracking=True),
                                  MaxPool1d(2, 2, tracking=True),
                                  Linker(permuting=True),
                                  BiLSTM(hidden_dim, hidden_dim, using_sequence=False),
                                  nn.Dropout(),
                                  nn.Linear(in_features=2 * hidden_dim, out_features=num_classes))

        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score = self._ops(x)
        return score

    def _init_weights(self, layer) -> None:
        if isinstance(layer, nn.Conv1d):
            nn.init.kaiming_uniform_(layer.weight)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
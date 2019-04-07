import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from typing import Dict


class Permute(nn.Module):
    """Permute class"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 1)


class ConvRec(nn.Module):
    """ConvRec class"""
    def __init__(self, num_classes: int, embedding_dim: int, hidden_dim: int, dic: Dict[str, int]) -> None:
        """Instantiating ConvRec class

        Args:
            num_classes (int): number of classes
            embedding_dim (int) : embedding dimension of token
            dic (dict): token2idx
        """
        super(ConvRec, self).__init__()
        self._extractor = nn.Sequential(nn.Embedding(len(dic), embedding_dim=embedding_dim, padding_idx=0),
                                        Permute(),
                                        nn.Conv1d(in_channels=embedding_dim, out_channels=hidden_dim, kernel_size=5),
                                        nn.ReLU(),
                                        nn.MaxPool1d(2, 2),
                                        nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3),
                                        nn.ReLU(),
                                        nn.MaxPool1d(2, 2),
                                        Permute())
        self._lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True,
                             bidirectional=True)
        self._fc = nn.Linear(in_features=2 * hidden_dim, out_features=num_classes)
        self._drop = nn.Dropout()
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor, x_len: torch.Tensor) -> torch.Tensor:
        fmap = self._extractor(x)
        fmap = pack_padded_sequence(fmap, lengths=self._get_length(x_len), batch_first=True)
        _, hc = self._lstm(fmap)
        feature = torch.cat([*hc[0]], dim=1)
        feature = self._drop(feature)
        score = self._fc(feature)
        return score

    def _init_weights(self, layer) -> None:
        if isinstance(layer, nn.Conv1d):
            nn.init.kaiming_uniform_(layer.weight)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)

    def _get_length(self, length: torch.Tensor) -> torch.Tensor:
        return (((length - 5 + 1) / 2) - 3 + 1) / 2
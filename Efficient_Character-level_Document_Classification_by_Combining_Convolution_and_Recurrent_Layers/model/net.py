import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

class ConvRec(nn.Module):
    """ConvRec class"""
    def __init__(self, num_classes: int, embedding_dim: int,
                 hidden_dim: int, dic: dict) -> None:
        """Instantiating ConvRec class

        Args:
            num_classes (int): number of classes
            embedding_dim (int) : embedding dimension of token
            dic (dict): token2idx
        """
        super(ConvRec, self).__init__()
        self._embedding = nn.Embedding(len(dic), embedding_dim=embedding_dim, padding_idx=0)
        self._conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=hidden_dim, kernel_size=5)
        self._conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3)
        self._lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True,
                             bidirectional=True)
        self._fc = nn.Linear(in_features=2 * hidden_dim, out_features=num_classes)

        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor, x_len: torch.Tensor, p=.5) -> torch.Tensor:
        x_batch = self._embedding(x)
        x_batch = x_batch.permute(0, 2, 1) # convert to (batch_size, channel, seq_len)
        fmap = F.relu(self._conv1(x_batch))
        fmap = F.max_pool1d(fmap, 2, 2)
        fmap = F.relu(self._conv2(fmap))
        fmap = F.max_pool1d(fmap, 2, 2)
        fmap = fmap.permute(0, 2, 1) # convert to (batch_size, seq_len, channel)
        fmap = pack_padded_sequence(fmap, lengths=self._get_length(x_len), batch_first=True)
        _, hc = self._lstm(fmap)
        flattend = torch.cat([*hc[0]], dim=1)
        flattend = F.dropout(flattend, p=p, training=self.training)
        score = self._fc(flattend)
        return score

    def _init_weights(self, layer) -> None:
        if isinstance(layer, nn.Conv1d):
            nn.init.kaiming_uniform_(layer.weight)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)

    def _get_length(self, length: torch.Tensor) -> torch.Tensor:
        return (((length - 5 + 1) / 2) - 3 + 1) / 2
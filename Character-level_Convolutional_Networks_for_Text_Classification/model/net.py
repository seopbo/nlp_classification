import torch
import torch.nn as nn
import torch.nn.functional as F

class CharCNN(nn.Module):
    """CharCNN class"""
    def __init__(self, num_classes: int, embedding_dim: int, dic: dict) -> None:
        """Instantiating CharCNN

        Args:
            num_classes (int): number of classes
            embedding_dim (int): embedding dimension of token
            dic (dict): token2idx
        """
        super(CharCNN, self).__init__()
        self._embedding = nn.Embedding(len(dic), embedding_dim, padding_idx=0)
        self._conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=256, kernel_size=7)
        self._conv2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=7)
        self._conv3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3)
        self._conv4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3)
        self._conv5 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3)
        self._conv6 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3)

        self._fc1 = nn.Linear(in_features=1792, out_features=512)
        self._fc2 = nn.Linear(in_features=512, out_features=512)
        self._fc3 = nn.Linear(in_features=512, out_features=num_classes)

        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor, p=.5) -> torch.Tensor:
        x_batch = self._embedding(x)
        x_batch = x_batch.permute(0, 2, 1)
        fmap = F.max_pool1d(F.relu(self._conv1(x_batch)), 3, 3)
        fmap = F.max_pool1d(F.relu(self._conv2(fmap)), 3, 3)
        fmap = F.relu(self._conv3(fmap))
        fmap = F.relu(self._conv4(fmap))
        fmap = F.relu(self._conv5(fmap))
        fmap = F.max_pool1d(F.relu(self._conv6(fmap)), 3, 3)
        flatten = fmap.view(fmap.shape[0], -1)
        dense = F.dropout(F.relu(self._fc1(flatten)), p=p, training=self.training)
        dense = F.dropout(F.relu(self._fc2(dense)), p=p, training=self.training)
        score = self._fc3(dense)
        return score

    def _init_weights(self, layer) -> None:
        if isinstance(layer, nn.Conv1d):
            nn.init.kaiming_uniform_(layer.weight)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
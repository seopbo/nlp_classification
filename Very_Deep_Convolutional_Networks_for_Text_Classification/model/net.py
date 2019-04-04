import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    """Flatten class"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class Permute(nn.Module):
    """Permute class"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 1)


class ConvBlock(nn.Module):
    """ConvBlock class"""
    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Instantiating ConvBlock class

        Args:
            in_channels (int): in_channels of ConvBlock
            out_channels (int): out_channels of ConvBlock
        """
        super(ConvBlock, self).__init__()
        self._equality = (in_channels != out_channels)
        self._ops = nn.Sequential(nn.Conv1d(in_channels, out_channels, 3, 1, 1),
                                  nn.BatchNorm1d(out_channels),
                                  nn.ReLU(),
                                  nn.Conv1d(out_channels, out_channels, 3, 1, 1),
                                  nn.BatchNorm1d(out_channels),
                                  nn.ReLU())

        if self._equality:
            self._shortcut = nn.Sequential(nn.Conv1d(in_channels, out_channels, 1, 1),
                                           nn.BatchNorm1d(out_channels))
        self._bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self._shortcut(x) if self._equality else x
        fmap = self._ops(x) + shortcut
        fmap = F.relu(self._bn(fmap))
        return fmap


class VDCNN(nn.Module):
    """VDCNN class"""
    def __init__(self, num_classes: int, embedding_dim: int, k_max: int, dic: dict):
        """Instantiating VDCNN class

        Args:
            num_classes (int): number of classes
            embedding_dim (int): embedding dimension of token
            k_max (int): parameter of k-max pooling following last convolution block
            dic (dict): token2idx
        """
        super(VDCNN, self).__init__()
        self._extractor = nn.Sequential(nn.Embedding(len(dic), embedding_dim, 0),
                                        Permute(),
                                        nn.Conv1d(embedding_dim, 64, 3, 1, 1),
                                        ConvBlock(64, 64),
                                        ConvBlock(64, 64),
                                        nn.MaxPool1d(2, 2),
                                        ConvBlock(64, 128),
                                        ConvBlock(128, 128),
                                        nn.MaxPool1d(2, 2),
                                        ConvBlock(128, 256),
                                        ConvBlock(256, 256),
                                        nn.MaxPool1d(2, 2),
                                        ConvBlock(256, 512),
                                        ConvBlock(512, 512),
                                        nn.AdaptiveMaxPool1d(k_max),
                                        Flatten())

        self._classifier = nn.Sequential(nn.Linear(512 * k_max, 2048),
                                         nn.ReLU(),
                                         nn.Linear(2048, 2048),
                                         nn.ReLU(),
                                         nn.Linear(2048, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = self._extractor(x)
        score = self._classifier(feature)
        return score


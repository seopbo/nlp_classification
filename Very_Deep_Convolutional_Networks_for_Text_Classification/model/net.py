import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """ConvBlock class"""
    def __init__(self, in_channels: int, out_channels: int, use_shortcut: bool) -> None:
        """Instantiating ConvBlock class

        Args:
            in_channels (int): in_channels of ConvBlock
            out_channels (int): out_channels of ConvBlock
            use_shortcut (bool): using identity shortcut if in_channels == out_channels -> True
        """
        super(ConvBlock, self).__init__()
        self._use_shortcut = use_shortcut
        self._equality = (in_channels != out_channels)
        self._ops = nn.Sequential(nn.Conv1d(in_channels, out_channels, 3, 1, 1),
                                            nn.BatchNorm1d(out_channels),
                                            nn.ReLU(),
                                            nn.Conv1d(out_channels, out_channels, 3, 1, 1),
                                            nn.BatchNorm1d(out_channels),
                                            nn.ReLU())

        if self._equality:
            self._shortcut = nn.Conv1d(in_channels, out_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self._use_shortcut:
            shortcut = self._shortcut(x) if self._equality else x
            fmap = self._ops(x) + shortcut
        else:
            fmap = self._ops(x)

        return fmap

class Flatten(nn.Module):
    """Flatten class"""
    def forward(self, x):
        return x.view(x.size(0), -1)

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
        self._embedding = nn.Embedding(len(dic), embedding_dim, 0)
        self._ops = nn.Sequential(nn.Conv1d(embedding_dim, 64, 3, 1, 1),
                                  ConvBlock(64, 64, True),
                                  ConvBlock(64, 64, True),
                                  nn.MaxPool1d(2, 2),
                                  ConvBlock(64, 128, True),
                                  ConvBlock(128, 128, True),
                                  nn.MaxPool1d(2, 2),
                                  ConvBlock(128, 256, True),
                                  ConvBlock(256, 256, True),
                                  nn.MaxPool1d(2, 2),
                                  ConvBlock(256, 512, True),
                                  ConvBlock(512, 512, True),
                                  nn.AdaptiveMaxPool1d(k_max),
                                  Flatten(),
                                  nn.Linear(4096, 2048),
                                  nn.ReLU(),
                                  nn.Dropout(),
                                  nn.Linear(2048, 2048),
                                  nn.ReLU(),
                                  nn.Dropout(),
                                  nn.Linear(2048, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_batch = self._embedding(x)
        x_batch = x_batch.permute(0, 2, 1)
        score = self._ops(x_batch)
        return score


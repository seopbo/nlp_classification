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

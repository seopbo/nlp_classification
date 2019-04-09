import torch
import torch.nn as nn


class Flatten(nn.Module):
    """Flatten class"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class Permute(nn.Module):
    """Permute class"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 1)

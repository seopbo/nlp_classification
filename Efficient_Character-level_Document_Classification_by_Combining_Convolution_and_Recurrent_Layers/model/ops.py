import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from typing import Tuple, Union, Callable


class Embedding(nn.Module):
    """Embedding class"""

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int = 0, permuting: bool = True,
                 tracking: bool = True) -> None:
        """Instantiating Embedding class

        Args:
            num_embeddings (int): the number of vocabulary size
            embedding_dim (int): the dimension of embedding vector
            padding_idx (int): denote padding_idx to "<pad>" token
            permuting (bool): permuting (n, l, c) -> (n, c, l). Default: True
            tracking (bool): tracking length of sequence. Default: True
        """
        super(Embedding, self).__init__()
        self._tracking = tracking
        self._permuting = permuting
        self._padding_idx = padding_idx
        self._ops = nn.Embedding(num_embeddings, embedding_dim, self._padding_idx)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        fmap = self._ops(x).permute(0, 2, 1) if self._permuting else self._ops(x)

        if self._tracking:
            fmap_length = x.ne(self._padding_idx).sum(dim=1)
            return fmap, fmap_length
        else:
            return fmap


class MaxPool1d(nn.Module):
    """MaxPool1d class"""

    def __init__(self, kernel_size: int, stride: int, tracking: bool = True) -> None:
        """Instantiating MaxPool1d class

        Args:
            kernel_size (int): the kernel size of 1d max pooling
            stride (int): the stride of 1d max pooling
            tracking (bool): tracking length of sequence. Default: True
        """
        super(MaxPool1d, self).__init__()
        self._kernel_size = kernel_size
        self._stride = stride
        self._tracking = tracking

        self._ops = nn.MaxPool1d(self._kernel_size, self._stride)

    def forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self._tracking:
            fmap, fmap_length = x
            fmap = self._ops(fmap)
            fmap_length = (fmap_length - (self._kernel_size - 1) - 1) / self._stride + 1
            return fmap, fmap_length
        else:
            fmap = self._ops(x)
            return fmap


class Conv1d(nn.Module):
    """Conv1dLayer class"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0,
                 activation: Callable[[torch.Tensor], torch.Tensor] = F.relu, tracking: bool = True) -> None:
        """Instantiating Conv1dLayer class

        Args:
            in_channels (int): the number of channels in the input feature map
            out_channels (int): the number of channels in the output feature emap
            kernel_size (int): the size of the convolving kernel
            stride (int): stride of the convolution. Default: 1
            padding (int): zero-padding added to both sides of the input. Default: 0
            activation (function): activation function. Default: F.relu
            tracking (bool): tracking length of sequence. Default: True
        """
        super(Conv1d, self).__init__()
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._ops = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self._activation = activation
        self._tracking = tracking

    def forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self._tracking:
            fmap, fmap_length = x
            fmap_length = (fmap_length + 2 * self._padding - (self._kernel_size - 1) - 1) / self._stride + 1
            fmap = self._activation(self._ops(fmap)) if self._activation is not None else self._ops(fmap)
            return fmap, fmap_length
        else:
            fmap = self._activation(self._ops(x)) if self._activation is not None else self._ops(x)
            return fmap


class Linker(nn.Module):
    """Linker class"""

    def __init__(self, permuting: bool = True) -> None:
        """Instantiating Linker class

        Args:
            permuting (bool): permuting (n, c, l) -> (n, l, c). Default: True
        """
        super(Linker, self).__init__()
        self._permuting = permuting

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> PackedSequence:
        fmap, fmap_length = x
        fmap = fmap.permute(0, 2, 1) if self._permuting else fmap
        return pack_padded_sequence(fmap, fmap_length, batch_first=True, enforce_sorted=False)


class BiLSTM(nn.Module):
    """BiLSTM class"""

    def __init__(self, input_size: int, hidden_size: int, using_sequence: bool = True) -> None:
        """Instantiating BiLSTM class""

        Args:
            input_size (int): the number of expected features in the input x
            hidden_size (int): the number of features in the hidden state h
            using_sequence (bool): using all hidden states of sequence. Default: True
        """
        super(BiLSTM, self).__init__()
        self._using_sequence = using_sequence
        self._ops = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, x: PackedSequence) -> torch.Tensor:
        outputs, hc = self._ops(x)

        if self._using_sequence:
            hiddens = pad_packed_sequence(outputs)[0].permute(1, 0, 2)
            return hiddens
        else:
            feature = torch.cat([*hc[0]], dim=1)
            return feature

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence
from model.utils import Vocab
from typing import Tuple, Union


class Embedding(nn.Module):
    """Embedding class"""

    def __init__(
        self,
        vocab: Vocab,
        padding_idx: int = 1,
        freeze: bool = True,
        permuting: bool = True,
        tracking: bool = True,
    ) -> None:
        """Instantiating Embedding class
        Args:
            vocab (model.utilsVocab): the instance of model.utils.Vocab
            padding_idx (int): denote padding_idx to padding token
            freeze (bool): freezing weights. Default: False
            permuting (bool): permuting (n, l, c) -> (n, c, l). Default: True
            tracking (bool): tracking length of sequence. Default: True
        """
        super(Embedding, self).__init__()
        self._padding_idx = padding_idx
        self._permuting = permuting
        self._tracking = tracking
        self._ops = nn.Embedding.from_pretrained(
            torch.from_numpy(vocab.embedding),
            freeze=freeze,
            padding_idx=self._padding_idx,
        )

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        fmap = self._ops(x).permute(0, 2, 1) if self._permuting else self._ops(x)

        if self._tracking:
            fmap_length = x.ne(self._padding_idx).sum(dim=1)
            return fmap, fmap_length
        else:
            return fmap


class Linker(nn.Module):
    """Linker class"""

    def __init__(self, permuting: bool = True):
        """Instantiating Linker class

        Args:
            permuting (bool): permuting (n, c, l) -> (n, l, c). Default: True
        """
        super(Linker, self).__init__()
        self._permuting = permuting

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> PackedSequence:
        fmap, fmap_length = x
        fmap = fmap.permute(0, 2, 1) if self._permuting else fmap
        return pack_padded_sequence(
            fmap, fmap_length, batch_first=True, enforce_sorted=False
        )


class LSTMEncoder(nn.Module):
    """LSTMEncoder class"""

    def __init__(
        self, input_size: int, hidden_size: int, using_sequence: bool = True
    ) -> None:
        """Instantiating LSTMEncoder class
        Args:
            input_size (int): the number of expected features in the input x
            hidden_size (int): the number of features in the hidden state h
            using_sequence (bool): using all hidden states of sequence. Default: True
        """
        super(LSTMEncoder, self).__init__()
        self._using_sequence = using_sequence
        self._ops = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x: PackedSequence) -> torch.Tensor:
        outputs, hc = self._ops(x)

        if self._using_sequence:
            hiddens = pad_packed_sequence(outputs, batch_first=True)[0]
            return hiddens
        else:
            feature = torch.cat([*hc[0]], dim=-1)
            return feature

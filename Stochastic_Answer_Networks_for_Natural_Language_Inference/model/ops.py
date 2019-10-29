import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence
from model.utils import Vocab
from typing import Tuple, Union, Callable


class PreEmbedding(nn.Module):
    """PreEmbedding class"""

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
            vocab (model.utils.Vocab): the instance of model.utils.Vocab
            padding_idx (int): denote padding_idx to padding token
            freeze (bool): freezing weights. Default: False
            permuting (bool): permuting (n, l, c) -> (n, c, l). Default: True
            tracking (bool): tracking length of sequence. Default: True
        """
        super(PreEmbedding, self).__init__()
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


class Conv1d(nn.Module):
    """Conv1d class"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 1,
                 activation: Callable[[torch.Tensor], torch.Tensor] = F.relu, tracking: bool = True) -> None:
        """Instantiating Conv1d class
        Args:
            in_channels (int): the number of channels in the input feature map
            out_channels (int): the number of channels in the output feature emap
            kernel_size (int): the size of the convolving kernel
            stride (int): stride of the convolution. Default: 1
            padding (int): zero-padding added to both sides of the input. Default: 1
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

import pickle
from torch.utils.data import DataLoader
from model.split import split_morphs, split_jamos
from model.utils import PreProcessor
from model.data import Corpus, batchify

with open('data/jamo_vocab.pkl', mode='rb') as io:
    jamo_vocab = pickle.load(io)
with open('data/morph_vocab.pkl', mode='rb') as io:
    morph_vocab = pickle.load(io)


preprocessor = PreProcessor(coarse_vocab=morph_vocab, fine_vocab=jamo_vocab,
                            coarse_split_fn=split_morphs, fine_split_fn=split_jamos)
ds = Corpus('data/train.txt', transform_fn=preprocessor.preprocess)
dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=batchify)

qa_mb, qb_mb, y_mb = next(iter(dl))

coarse_emb = PreEmbedding(morph_vocab, padding_idx=morph_vocab.to_indices(morph_vocab.padding_token),
                          freeze=False, permuting=False)
fine_emb = nn.Embedding(num_embeddings=len(jamo_vocab), embedding_dim=32,
                        padding_idx=jamo_vocab.to_indices(jamo_vocab.padding_token))
coarse_emb(qa_mb[0])[0].shape
qa_mb[1].shape
tst = fine_emb(qa_mb[1])
tst.shape
ops1=nn.Conv2d(in_channels=10, out_channels=50, kernel_size=1, groups=2)
ops1(tst)
torch.unbind(tst, 0)[0]

class LexiconEncoder(nn.Module):
    def __init__(self, coarse_vocab, fine_vocab):
        super(LexiconEncoder, self).__init__()
        self._coarse_emb = PreEmbedding(coarse_vocab, padding_idx=coarse_vocab.to_indices(coarse_vocab.padding_token),
                                        freeze=False, permuting=False)
        self._fine_emb = PreEmbedding(fine_vocab, padding_idx=fine_vocab.to_indices(fine_vocab.padding_token),
                                      freeze=False, permuting=True)

torch.nn.LayerNorm
pytorch old version
{'ops.LayerNorm.gamma'}
import pickle
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence, pad_packed_sequence
from torch.utils.data import DataLoader
from typing import Union, Tuple
from model.utils import Vocab, Tokenizer, TeacherForcing
from model.data import batchify, NMTCorpus
from model.split import split_morphs, split_space

with open('data/vocab_ko.pkl', mode='rb') as io:
    vocab_ko = pickle.load(io)
ko_processor = Tokenizer(vocab_ko, split_morphs)
with open('data/vocab_en.pkl', mode='rb') as io:
    vocab_en = pickle.load(io)
en_processor = TeacherForcing(vocab_en, split_space)

ds = NMTCorpus('data/train.txt', ko_processor.split_and_transform, en_processor.process)
dl = DataLoader(ds, 2, shuffle=True, num_workers=4, collate_fn=batchify)
x, y1, y2 = next(iter(dl))


class Embedding(nn.Module):
    """Embedding class"""
    def __init__(self, vocab: Vocab, padding_idx: int = 1, freeze: bool = True,
                 permuting: bool = True, tracking: bool = True) -> None:
        """Instantiating Embedding class
        Args:
            vocab (model.utils.Vocab): the instance of model.utils.Vocab
            padding_idx (int): denote padding_idx to padding token
            freeze (bool): freezing weights. Default: False
            permuting (bool): permuting (n, l, c) -> (n, c, l). Default: True
            tracking (bool): tracking length of sequence. Default: True
        """
        super(Embedding, self).__init__()
        self._padding_idx = padding_idx
        self._permuting = permuting
        self._tracking = tracking
        self._ops = nn.Embedding.from_pretrained(torch.from_numpy(vocab.embedding),
                                                 freeze=freeze, padding_idx=self._padding_idx)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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
        return pack_padded_sequence(fmap, fmap_length, batch_first=True, enforce_sorted=False)


class Encoder(nn.Module):
    """Encoder class"""
    def __init__(self, input_size: int, hidden_size: int, using_sequence: bool = True) -> None:
        """Instantiating Encoder class
        Args:
            input_size (int): the number of expected features in the input x
            hidden_size (int): the number of features in the hidden state h
            using_sequence (bool): using all hidden states of sequence. Default: True
        """
        super(Encoder, self).__init__()
        self._using_sequence = using_sequence
        self._ops = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=2)

    def forward(self, x: PackedSequence) -> torch.Tensor:
        outputs, hc = self._ops(x)

        if self._using_sequence:
            hiddens = pad_packed_sequence(outputs)[0].permute(1, 0, 2)
            return hiddens
        else:
            feature = torch.cat([*hc[0]], dim=1)
            return feature


class AttnDecoder(nn.Module):
    def __init__(self):
        super(AttnDecoder, self).__init__()



    def forward(self, ):
ko_emb = Embedding(ko_processor.vocab, padding_idx=1, freeze=False, permuting=False, tracking=True)
en_emb = Embedding(en_processor.vocab, padding_idx=1, freeze=False, permuting=False, tracking=True)
linker = Linker(permuting=False)

x_mb = linker(ko_emb(x))
encoder = Encoder(input_size=300, hidden_size=128, using_sequence=True)
encoder_outputs = encoder(x_mb)

y_mb = linker(en_emb(y1))
decoder_outputs = encoder(y_mb)

encoder_outputs.shape

encoder_outputs.shape
decoder_outputs.unsqueeze(1).shape
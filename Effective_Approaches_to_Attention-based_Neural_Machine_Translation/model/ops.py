import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence, pad_packed_sequence
from torch.utils.data import DataLoader
from typing import Union, Tuple
from model.utils import Vocab, SourceProcessor, TargetProcessor
from model.data import batchify, NMTCorpus
from model.split import split_morphs, split_space


with open('data/vocab_ko.pkl', mode='rb') as io:
    vocab_ko = pickle.load(io)
ko_processor = SourceProcessor(vocab_ko, split_morphs)
with open('data/vocab_en.pkl', mode='rb') as io:
    vocab_en = pickle.load(io)
en_processor = TargetProcessor(vocab_en, split_space)

ds = NMTCorpus('data/train.txt', ko_processor.process, en_processor.process)
dl = DataLoader(ds, 2, shuffle=False, num_workers=4, collate_fn=batchify)
x, y = next(iter(dl))


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
            permuting (bool): permuting (0, 1, 2) -> (0, 2, 1). Default: True
        """
        super(Linker, self).__init__()
        self._permuting = permuting

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> PackedSequence:
        fmap, fmap_length = x
        fmap = fmap.permute(0, 2, 1) if self._permuting else fmap
        return pack_padded_sequence(fmap, fmap_length, batch_first=True, enforce_sorted=False)


class Encoder(nn.Module):
    """Encoder class"""
    def __init__(self, input_size: int, hidden_size: int, vocab: Vocab) -> None:
        """Instantiating Encoder class

        Args:
            input_size (int): the number of expected features in the input x
            hidden_size (int): the number of features in the hidden state h
        """
        super(Encoder, self).__init__()
        self._emb = Embedding(vocab=vocab, padding_idx=vocab.to_indices(vocab.padding_token), freeze=False,
                              permuting=False, tracking=True)
        self._linker = Linker(permuting=False)
        self._ops = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=2)

    def forward(self, x: torch.Tensor, hc: Tuple[torch.Tensor, torch.Tensor] = None) ->\
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        embed = self._emb(x)
        packed_embed = self._linker(embed)
        ops_outputs, hc = self._ops(packed_embed, hc)
        ops_outputs, _ = pad_packed_sequence(ops_outputs, batch_first=True)
        return ops_outputs, hc


# class Attn(nn.Module):
#     def __init__(self, method, hidden_size):
#         super(Attn, self).__init__()
#         self.method = method
#         if self.method not in ['dot', 'general', 'concat']:
#             raise ValueError(self.method, "is not an appropriate attention method.")
#         self.hidden_size = hidden_size
#         if self.method == 'general':
#             self.attn = nn.Linear(self.hidden_size, hidden_size)
#         elif self.method == 'concat':
#             self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
#             self.v = nn.Parameter(torch.FloatTensor(hidden_size))
#
#     def dot_score(self, hidden, encoder_output):
#         return torch.sum(hidden * encoder_output, dim=2)
#
#     def general_score(self, hidden, encoder_output):
#         energy = self.attn(encoder_output)
#         return torch.sum(hidden * energy, dim=2)
#
#     def concat_score(self, hidden, encoder_output):
#         energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
#         return torch.sum(self.v * energy, dim=2)
#
#     def forward(self, hidden, encoder_outputs):
#         # Calculate the attention weights (energies) based on the given method
#         if self.method == 'general':
#             attn_energies = self.general_score(hidden, encoder_outputs)
#         elif self.method == 'concat':
#             attn_energies = self.concat_score(hidden, encoder_outputs)
#         elif self.method == 'dot':
#             attn_energies = self.dot_score(hidden, encoder_outputs)
#
#         # Transpose max_length and batch_size dimensions
#         attn_energies = attn_energies.t()
#
#         # Return the softmax normalized probability scores (with added dimension)
#         return F.softmax(attn_energies, dim=1).unsqueeze(1)


class Decoder(nn.Module):
    """Decoder class"""
    def __init__(self, input_size: int, hidden_size: int, vocab: Vocab) -> None:
        """Instantiating Encoder class

        Args:
            input_size (int): the number of expected features in the input x
            hidden_size (int): the number of features in the hidden state h
        """
        super(Decoder, self).__init__()
        self._emb = Embedding(vocab=vocab, padding_idx=vocab.to_indices(vocab.padding_token),
                              freeze=False, permuting=False, tracking=False)
        self._ops = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=2)
        self._classifier = nn.Linear(hidden_size, len(vocab))

    def forward(self, x, hc):
        embed = self._emb(x)
        ops_output, hc = self._ops(embed, hc)
        output = ops_output.squeeze(1)
        output = self._classifier(output)
        return ops_output, output, hc


encoder = Encoder(input_size=300, hidden_size=128, vocab=ko_processor.vocab)
encoder_outputs, encoder_hc = encoder(x)

decoder = Decoder(input_size=300, hidden_size=128, vocab=en_processor.vocab)
decoder_input = torch.LongTensor([vocab_en.to_indices(vocab_en.bos_token) for _ in range(2)]).reshape(-1,1)

ops_output, output, hc = decoder(decoder_input, encoder_hc)

y
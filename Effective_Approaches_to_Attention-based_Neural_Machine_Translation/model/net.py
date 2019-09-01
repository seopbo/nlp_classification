import torch
import torch.nn as nn
from typing import Tuple
from torch.nn.utils.rnn import pad_packed_sequence
from model.utils import Vocab
from model.ops import Embedding, Linker, GlobalAttn


class Encoder(nn.Module):
    """Encoder class"""
    def __init__(self, vocab: Vocab, encoder_hidden_dim: int, drop_ratio: int = .2) -> None:
        """Instantiating Encoder class

        Args:
            vocab (model.utils.Vocab): the instance of model.utils.Vocab
            encoder_hidden_dim (int): the dimension of hidden state and cell state
            drop_ratio (float): ratio of drop out
        """
        super(Encoder, self).__init__()
        self._emb = Embedding(vocab=vocab, padding_idx=vocab.to_indices(vocab.padding_token), freeze=False,
                              permuting=False, tracking=True)
        self._emb_dropout = nn.Dropout(drop_ratio)
        self._linker = Linker(permuting=False)
        self._ops = nn.LSTM(self._emb._ops.embedding_dim,
                            encoder_hidden_dim, batch_first=True, num_layers=4, dropout=drop_ratio)
        self._dropout = nn.Dropout(drop_ratio)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embed, source_length = self._emb(x)
        embed = self._emb_dropout(embed)
        packed_embed = self._linker((embed, source_length))
        encoder_outputs, hc = self._ops(packed_embed)
        encoder_outputs, _ = pad_packed_sequence(encoder_outputs, batch_first=True)
        encoder_outputs = self._dropout(encoder_outputs)
        return encoder_outputs, source_length, hc


class AttnDecoder(nn.Module):
    """AttnDecoder class"""
    def __init__(self, vocab: Vocab, method, encoder_hidden_dim: int, decoder_hidden_dim: int, drop_ratio: int = .2) -> None:
        """Instantiating Encoder class

        Args:
            input_dim (int): the number of expected features in the input x
            encoder_hidden_dim (int): the number of features in the hidden state h
        """
        super(AttnDecoder, self).__init__()
        self._emb = Embedding(vocab=vocab, padding_idx=vocab.to_indices(vocab.padding_token),
                              freeze=False, permuting=False, tracking=False)
        self._emb_dropout = nn.Dropout(drop_ratio)
        self._ops = nn.LSTM(self._emb._ops.embedding_dim,
                            decoder_hidden_dim, batch_first=True, num_layers=4, dropout=drop_ratio)
        self._dropout = nn.Dropout(drop_ratio)
        self._attn = GlobalAttn(method=method, encoder_hidden_dim=encoder_hidden_dim,
                                decoder_hidden_dim=decoder_hidden_dim)
        self._concat = nn.Linear(encoder_hidden_dim + decoder_hidden_dim, self._emb._ops.embedding_dim, bias=False)

    def forward(self, x, hc, encoder_outputs, source_length):
        embed = self._emb(x)
        embed = self._emb_dropout(embed)
        ops_output, hc = self._ops(embed, hc)
        ops_output = self._dropout(ops_output)
        context = self._attn(ops_output, encoder_outputs, source_length)
        ops_output = ops_output.squeeze(1)
        output = torch.tanh(self._concat(torch.cat([ops_output, context], dim=-1)))
        decoder_output = output @ self._emb._ops.weight.t()
        return decoder_output, hc

import pickle
from model.data import NMTCorpus
from torch.utils.data import DataLoader
from model.data import batchify
from model.utils import SourceProcessor, TargetProcessor
from model.split import Stemmer

with open('data/vocab_ko.pkl', mode='rb') as io:
    src_vocab = pickle.load(io)
src_stemmer = Stemmer(language='ko')
src_processor = SourceProcessor(src_vocab, src_stemmer.extract_stem)


with open('data/vocab_en.pkl', mode='rb') as io:
    tgt_vocab = pickle.load(io)
tgt_stemmer = Stemmer(language='en')
tgt_processor = TargetProcessor(tgt_vocab, tgt_stemmer.extract_stem)
ds = NMTCorpus('data/train.txt', src_processor.process, tgt_processor.process)
dl = DataLoader(ds, batch_size=2, collate_fn=batchify)
mb = next(iter(dl))


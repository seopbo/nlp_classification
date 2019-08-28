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
            input_size (int): the number of expected features in the input x
            encoder_hidden_dim (int): the number of features in the hidden state h
        """
        super(Encoder, self).__init__()
        self._emb = Embedding(vocab=vocab, padding_idx=vocab.to_indices(vocab.padding_token), freeze=False,
                              permuting=False, tracking=True)
        self._emb_dropout = nn.Dropout(drop_ratio)
        self._linker = Linker(permuting=False)
        self._ops = nn.LSTM(self._emb._ops.embedding_dim,
                            encoder_hidden_dim, batch_first=True, num_layers=2, dropout=drop_ratio)
        self._dropout = nn.Dropout(drop_ratio)

    def forward(self, x: torch.Tensor, hc: Tuple[torch.Tensor, torch.Tensor] = None) ->\
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        embed, source_length = self._emb(x)
        embed = self._emb_dropout(embed)
        packed_embed = self._linker((embed, source_length))
        encoder_outputs, hc = self._ops(packed_embed, hc)
        encoder_outputs, _ = pad_packed_sequence(encoder_outputs, batch_first=True)
        encoder_outputs = self._dropout(encoder_outputs)
        return encoder_outputs, hc, source_length


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
                            decoder_hidden_dim, batch_first=True, num_layers=2, dropout=drop_ratio)
        self._dropout = nn.Dropout(drop_ratio)
        self._attn = GlobalAttn(method=method, encoder_hidden_dim=encoder_hidden_dim,
                                decoder_hidden_dim=decoder_hidden_dim)
        self._classifier = nn.Linear(encoder_hidden_dim + decoder_hidden_dim, len(vocab))

    def forward(self, x, hc, encoder_outputs, source_length):
        embed = self._emb(x)
        embed = self._emb_dropout(embed)
        ops_output, hc = self._ops(embed, hc)
        ops_output = self._dropout(ops_output)
        context = self._attn(ops_output, encoder_outputs, source_length)
        ops_output = ops_output.squeeze(1)
        output = torch.cat([ops_output, context], dim=-1)
        decoder_output = self._classifier(output)
        return decoder_output, hc

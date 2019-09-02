import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence
from typing import Tuple
from model.utils import Vocab
from model.ops import Embedding, Linker, GlobalAttn


class Encoder(nn.Module):
    """Encoder class"""
    def __init__(self, vocab: Vocab, encoder_hidden_dim: int, drop_ratio: int = .2) -> None:
        """Instantiating Encoder class

        Args:
            vocab (model.utils.Vocab): the instance of model.utils.Vocab
            encoder_hidden_dim (int): the dimension of hidden state and cell state
            drop_ratio (float): ratio of drop out, default 0.2
        """
        super(Encoder, self).__init__()
        self._emb = Embedding(vocab=vocab, padding_idx=vocab.to_indices(vocab.padding_token), freeze=False,
                              permuting=False, tracking=True)
        self._linker = Linker(permuting=False)
        self._ops = nn.LSTM(self._emb._ops.embedding_dim,
                            encoder_hidden_dim, batch_first=True, num_layers=2, dropout=drop_ratio)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embed, source_length = self._emb(x)
        packed_embed = self._linker((embed, source_length))
        encoder_outputs, encoder_hc = self._ops(packed_embed)
        encoder_outputs, _ = pad_packed_sequence(encoder_outputs, batch_first=True)
        return encoder_outputs, source_length, encoder_hc


class AttnDecoder(nn.Module):
    """AttnDecoder class"""
    def __init__(self, vocab: Vocab, method: str, encoder_hidden_dim: int,
                 decoder_hidden_dim: int, drop_ratio: int = .2) -> None:
        """Instantiating Encoder class

        Args:
            vocab (model.utils.Vocab): the instance of model.utils.Vocab
            method (str): the method of attention, 'dot', 'general', 'concat'
            encoder_hidden_dim (int): the dimension of hidden state and cell state of encoder
            decoder_hidden_dim (int): the dimension of hidden state and cell state of decoder
            drop_ratio (float): ratio of drop out, default 0.2
        """
        super(AttnDecoder, self).__init__()
        self._emb = Embedding(vocab=vocab, padding_idx=vocab.to_indices(vocab.padding_token),
                              freeze=False, permuting=False, tracking=False)
        self._ops = nn.LSTM(self._emb._ops.embedding_dim, decoder_hidden_dim, batch_first=True,
                            num_layers=2, dropout=drop_ratio)
        self._attn = GlobalAttn(method=method, encoder_hidden_dim=encoder_hidden_dim,
                                decoder_hidden_dim=decoder_hidden_dim)
        self._concat = nn.Linear(encoder_hidden_dim + decoder_hidden_dim, self._emb._ops.embedding_dim, bias=False)
        # self._classify = nn.Linear(self._emb._ops.embedding_dim, len(vocab))

    def forward(self, x, hc, encoder_outputs, source_length):
        embed = self._emb(x)
        ops_output, hc = self._ops(embed, hc)
        context = self._attn(ops_output, encoder_outputs, source_length)
        ops_output = ops_output.squeeze()
        output = torch.tanh(self._concat(torch.cat([ops_output, context], dim=-1)))
        # decoder_output = self._classify(output)
        decoder_output = output @ self._emb._ops.weight.t()
        return decoder_output, hc

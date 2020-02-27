import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
from typing import Union, Tuple
from model.utils import Vocab


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


class GlobalAttn(nn.Module):
    def __init__(self, method, encoder_output_dim, decoder_hidden_dim):
        super(GlobalAttn, self).__init__()
        self._method = method
        self._encoder_output_dim = encoder_output_dim
        self._decoder_hidden_dim = decoder_hidden_dim

        if self._method == 'general':
            self._wa = nn.Parameter(torch.Tensor(encoder_output_dim, decoder_hidden_dim))
            nn.init.xavier_normal_(self._wa)
        elif self._method == 'concat':
            self._wa = nn.Parameter(torch.Tensor(encoder_output_dim + decoder_hidden_dim, 1))
            nn.init.xavier_normal_(self._wa)

        self._attn = {'dot': self._dot_score,
                      'general': self._general_score,
                      'concat': self._concat_score}

    def forward(self, decoder_output, encoder_outputs, source_length):
        attn_weights = self._attn[self._method](decoder_output, encoder_outputs, source_length)
        attn_weights = attn_weights.unsqueeze(-1)
        context = torch.bmm(encoder_outputs.permute(0, 2, 1), attn_weights).squeeze(-1)
        return context

    def _dot_score(self, decoder_output, encoder_outputs, source_length):
        attn_mask = self._generate_mask(source_length)
        score = torch.bmm(encoder_outputs, decoder_output.permute(0, 2, 1)).squeeze(-1)
        score[~attn_mask] = float('-inf')
        attn_weights = F.softmax(score, dim=-1)
        return attn_weights

    def _general_score(self, decoder_output, encoder_outputs, source_length):
        attn_mask = self._generate_mask(source_length)
        score = torch.bmm(encoder_outputs @ self._wa, decoder_output.permute(0, 2, 1)).squeeze(-1)
        score[~attn_mask] = float('-inf')
        attn_weights = F.softmax(score, dim=-1)
        return attn_weights

    def _concat_score(self, decoder_output, encoder_outputs, source_length):
        attn_mask = self._generate_mask(source_length)
        decoder_outputs = torch.cat([decoder_output for _ in range(encoder_outputs.size(1))], dim=1)
        outputs = torch.cat([encoder_outputs, decoder_outputs], dim=-1)
        score = (outputs @ self._wa).squeeze(-1)
        score[~attn_mask] = float('-inf')
        attn_weights = F.softmax(score, dim=-1)
        return attn_weights

    def _generate_mask(self, source_length):
        attn_mask = torch.arange(source_length.max(),
                                 device=source_length.device)[None, :] < source_length[:, None]
        return attn_mask

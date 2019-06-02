import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence
from gluonnlp import Vocab
from typing import Tuple, Union


class PreEmbedding(nn.Module):
    """PreEmbedding class"""
    def __init__(self, vocab: Vocab, padding_idx: int = 1, freeze: bool = True,
                 permuting: bool = True, tracking: bool = True) -> None:
        """Instantiating PreEmbedding class

        Args:
            vocab (gluonnlp.Vocab): the instance of gluonnlp.Vocab
            padding_idx (int): denote padding_idx to padding token
            freeze (bool): freezing weigths. Default: False
            permuting (bool): permuting (n, l, c) -> (n, c, l). Default: True
            tracking (bool): tracking length of sequence. Default: True
        """
        super(PreEmbedding, self).__init__()
        self._padding_idx = padding_idx
        self._permuting = permuting
        self._tracking = tracking
        self._ops = nn.Embedding.from_pretrained(torch.from_numpy(vocab.embedding.idx_to_vec.asnumpy()),
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


class BiLSTM(nn.Module):
    """BiLSTM class"""
    def __init__(self, input_size: int, hidden_size: int, using_sequence: bool = True) -> None:
        """Instantiating BiLSTM class

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

# torch.randn?
# class CRF(nn.Module):
#     def __init__(self, label_vocab):
#         super(CRF, self).__init__()
#         self._num_tags = len(label_vocab)
#
#         # matrix of transition scores from j to i
#         self.trans = nn.Parameter(torch.randn())
#         self.trans.data[SOS_IDX, :] = -10000. # no transition to SOS
#         self.trans.data[:, EOS_IDX] = -10000. # no transition from EOS except to PAD
#         self.trans.data[:, PAD_IDX] = -10000. # no transition from PAD except to PAD
#         self.trans.data[PAD_IDX, :] = -10000. # no transition to PAD except from EOS
#         self.trans.data[PAD_IDX, EOS_IDX] = 0.
#         self.trans.data[PAD_IDX, PAD_IDX] = 0.
#
#     def forward(self, h, mask): # forward algorithm
#         # initialize forward variables in log space
#         score = Tensor(BATCH_SIZE, self._num_tags).fill_(-10000.) # [B, C]
#         score[:, SOS_IDX] = 0.
#         trans = self.trans.unsqueeze(0) # [1, C, C]
#         for t in range(h.size(1)): # recursion through the sequence
#             mask_t = mask[:, t].unsqueeze(1)
#             emit_t = h[:, t].unsqueeze(2) # [B, C, 1]
#             score_t = score.unsqueeze(1) + emit_t + trans # [B, 1, C] -> [B, C, C]
#             score_t = log_sum_exp(score_t) # [B, C, C] -> [B, C]
#             score = score_t * mask_t + score * (1 - mask_t)
#         score = log_sum_exp(score + self.trans[EOS_IDX])
#         return score # partition function
#
#     def score(self, h, y, mask): # calculate the score of a given sequence
#         score = Tensor(BATCH_SIZE).fill_(0.)
#         h = h.unsqueeze(3)
#         trans = self.trans.unsqueeze(2)
#         for t in range(h.size(1)): # recursion through the sequence
#             mask_t = mask[:, t]
#             emit_t = torch.cat([h[t, y[t + 1]] for h, y in zip(h, y)])
#             trans_t = torch.cat([trans[y[t + 1], y[t]] for y in y])
#             score += (emit_t + trans_t) * mask_t
#         last_tag = y.gather(1, mask.sum(1).long().unsqueeze(1)).squeeze(1)
#         score += self.trans[EOS_IDX, last_tag]
#         return score
#
#     def decode(self, h, mask): # Viterbi decoding
#         # initialize backpointers and viterbi variables in log space
#         bptr = LongTensor()
#         score = Tensor(BATCH_SIZE, self._num_tags).fill_(-10000.)
#         score[:, SOS_IDX] = 0.
#
#         for t in range(h.size(1)): # recursion through the sequence
#             mask_t = mask[:, t].unsqueeze(1)
#             score_t = score.unsqueeze(1) + self.trans # [B, 1, C] -> [B, C, C]
#             score_t, bptr_t = score_t.max(2) # best previous scores and tags
#             score_t += h[:, t] # plus emission scores
#             bptr = torch.cat((bptr, bptr_t.unsqueeze(1)), 1)
#             score = score_t * mask_t + score * (1 - mask_t)
#         score += self.trans[EOS_IDX]
#         best_score, best_tag = torch.max(score, 1)
#
#         # back-tracking
#         bptr = bptr.tolist()
#         best_path = [[i] for i in best_tag.tolist()]
#         for b in range(BATCH_SIZE):
#             x = best_tag[b] # best tag
#             y = int(mask[b].sum().item())
#             for bptr_t in reversed(bptr[b][:y]):
#                 x = bptr_t[x]
#                 best_path[b].append(x)
#             best_path[b].pop()
#             best_path[b].reverse()
#
#         return best_path

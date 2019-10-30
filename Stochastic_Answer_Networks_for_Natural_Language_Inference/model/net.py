import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from model.ops import LexiconEncoder, ContextualEncoder, BiLSTM
from typing import Tuple


import pickle
from torch.utils.data import DataLoader
from model.split import split_morphs, split_jamos
from model.utils import PreProcessor
from model.data import Corpus, batchify

with open("data/jamo_vocab.pkl", mode="rb") as io:
    jamo_vocab = pickle.load(io)
with open("data/morph_vocab.pkl", mode="rb") as io:
    morph_vocab = pickle.load(io)


preprocessor = PreProcessor(
    coarse_vocab=morph_vocab,
    fine_vocab=jamo_vocab,
    coarse_split_fn=split_morphs,
    fine_split_fn=split_jamos,
)
ds = Corpus("data/train.txt", transform_fn=preprocessor.preprocess)
dl = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=batchify)

qa_mb, qb_mb, y_mb = next(iter(dl))


class SAN(nn.Module):
    def __init__(self, coarse_vocab, fine_vocab, fine_embedding_dim, hidden_size):
        super(SAN, self).__init__()

        self._lenc = LexiconEncoder(coarse_vocab, fine_vocab, fine_embedding_dim)
        self._cenc = ContextualEncoder(self._lenc._output_size, hidden_size)
        self._proj = nn.Linear(hidden_size * 2, hidden_size * 2)
        self._drop_a = nn.Dropout()
        self._drop_b = nn.Dropout()
        self._bilstm = BiLSTM(input_size=6*hidden_size, hidden_size=hidden_size, using_sequence=True)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        qa_mb, qb_mb = inputs

        # encoding
        ca, length_a = self._cenc(self._lenc(qa_mb))
        cb, length_b = self._cenc(self._lenc(qb_mb))

        # attention
        proj_ca = F.relu(self._proj(ca))
        proj_cb = F.relu(self._proj(cb))

        # for a
        attn_score_a = torch.bmm(proj_ca, proj_cb.permute(0, 2, 1))
        attn_score_a = self._drop_a(attn_score_a)
        attn_a = F.softmax(attn_score_a, dim=-1)

        # for b
        attn_score_b = torch.bmm(proj_cb, proj_ca.permute(0, 2, 1))
        attn_score_b = self._drop_b(attn_score_b)
        attn_b = F.softmax(attn_score_b, dim=-1)

        # memory
        ua = torch.cat([proj_ca, torch.bmm(attn_a, proj_cb)], dim=-1)
        ub = torch.cat([proj_cb, torch.bmm(attn_b, proj_ca)], dim=-1)
        feature_a = pack_padded_sequence(torch.cat([ua, ca], dim=-1), length_a, batch_first=True)
        feature_b = pack_padded_sequence(torch.cat([ub, cb], dim=-1), length_b, batch_first=True)
        ma = self._bilstm(feature_a)
        mb = self._bilstm(feature_b)

        # answer
        return ma, mb

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.nn.utils.rnn import pack_padded_sequence
from model.ops import LexiconEncoder, ContextualEncoder, BiLSTM
from typing import Tuple


class SAN(nn.Module):
    def __init__(self, num_classes, coarse_vocab, fine_vocab, fine_embedding_dim, hidden_size, multi_step,
                 prediction_drop_ratio):
        super(SAN, self).__init__()

        self._lenc = LexiconEncoder(coarse_vocab, fine_vocab, fine_embedding_dim)
        self._cenc = ContextualEncoder(self._lenc._output_size, hidden_size)
        self._proj = nn.Linear(hidden_size * 2, hidden_size * 2, bias=False)
        self._drop_a = nn.Dropout(.2)
        self._drop_b = nn.Dropout(.2)
        self._bilstm = BiLSTM(input_size=6 * hidden_size, hidden_size=hidden_size, using_sequence=True)
        self._theta_a = nn.Linear(2 * hidden_size, 1, bias=False)
        self._theta_b = nn.Parameter(torch.randn(2 * hidden_size, 2 * hidden_size))
        self._grucell = nn.GRUCell(2 * hidden_size, 2 * hidden_size)
        self._prediction = nn.Linear(8 * hidden_size, num_classes)
        self._multi_step = multi_step
        self._prediction_drop_ratio = prediction_drop_ratio

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
        ua = torch.cat([ca, torch.bmm(attn_a, cb)], dim=-1)
        ub = torch.cat([cb, torch.bmm(attn_b, ca)], dim=-1)
        feature_a = pack_padded_sequence(torch.cat([ua, ca], dim=-1), length_a, batch_first=True, enforce_sorted=False)
        feature_b = pack_padded_sequence(torch.cat([ub, cb], dim=-1), length_b, batch_first=True, enforce_sorted=False)
        ma = self._bilstm(feature_a)
        mb = self._bilstm(feature_b)

        # answer
        weights_alpha = torch.softmax(self._theta_a(ma).permute(0, 2, 1), dim=-1)
        hidden_state = torch.bmm(weights_alpha, ma).squeeze()
        weights_beta = torch.softmax((hidden_state.unsqueeze(1) @ self._theta_b @ mb.permute(0, 2, 1)), dim=-1)
        time_step_input = torch.bmm(weights_beta, mb).squeeze()

        predictions = []
        predictions.append(self._one_step_predict((hidden_state, time_step_input)))

        for step in range(self._multi_step - 1):
            hidden_state = self._grucell(time_step_input, hidden_state)
            weights_beta = torch.softmax((hidden_state.unsqueeze(1) @ self._theta_b @ mb.permute(0, 2, 1)), dim=-1)
            time_step_input = torch.bmm(weights_beta, mb).squeeze()

            predictions.append(self._one_step_predict((hidden_state, time_step_input)))
        else:
            predictions = torch.stack(predictions)

            if self.training:
                selected_indices = torch.where(torch.rand(self._multi_step).ge(self._prediction_drop_ratio))[0]
                selected_indices = selected_indices.to(time_step_input.device)
                average_prediction = predictions.index_select(0, selected_indices).mean(0)
            else:
                average_prediction = predictions.mean(0)

        return average_prediction

    def _one_step_predict(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        hidden_state, time_step_input = x
        concatenated = torch.cat([hidden_state, time_step_input, torch.abs(hidden_state - time_step_input),
                                  hidden_state * time_step_input], dim=-1)
        prediction = torch.softmax(self._prediction(concatenated), dim=-1)
        return prediction

#
# import pickle
# from torch.utils.qpair import DataLoader
# from model.split import split_morphs, split_jamos
# from model.utils import PreProcessor
# from model.qpair import Corpus, batchify
#
# with open("qpair/jamo_vocab.pkl", mode="rb") as io:
#     jamo_vocab = pickle.load(io)
# with open("qpair/morph_vocab.pkl", mode="rb") as io:
#     morph_vocab = pickle.load(io)
#
#
# preprocessor = PreProcessor(
#     coarse_vocab=morph_vocab,
#     fine_vocab=jamo_vocab,
#     coarse_split_fn=split_morphs,
#     fine_split_fn=split_jamos,
# )
# ds = Corpus("qpair/train.txt", transform_fn=preprocessor.preprocess)
# dl = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=batchify)
#
# qa_mb, qb_mb, y_mb = next(iter(dl))
# model = SAN(2, morph_vocab, jamo_vocab, 32, 128, multi_step=5)
# model.eval()
# prediction = model((qa_mb, qb_mb))
# prediction
# loss = nn.NLLLoss()
# torch.log(prediction)
#
# loss(torch.log(prediction), y_mb)
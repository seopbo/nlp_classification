import torch
import torch.nn as nn
from model.ops import PreEmbedding, Linker, BiLSTM, CRF
from gluonnlp import Vocab
from typing import Tuple


class BilstmCRF(nn.Module):
    """BilstmCRF"""
    def __init__(self, label_vocab: Vocab, token_vocab: Vocab, lstm_hidden_dim: int) -> None:
        """Instantiating BilstmCRF class

        Args:
            token_vocab: (gluonnlp.Vocab): the instance of gluonnlp.Vocab that has token information
            label_vocab: (gluonnlp.Vocab): the instance of gluonnlp.Vocab that has label information
            lstm_hidden_dim (int): the number of hidden dimension of lstm
        """
        super(BilstmCRF, self).__init__()
        self._embedding = PreEmbedding(token_vocab, padding_idx=token_vocab.to_indices(token_vocab.padding_token),
                                       freeze=False, permuting=False, tracking=True)
        self._pipe = Linker(permuting=False)
        self._bilstm = BiLSTM(self._embedding._ops.embedding_dim, lstm_hidden_dim, using_sequence=True)
        self._fc = nn.Linear(2 * lstm_hidden_dim, len(label_vocab))
        self._crf = CRF(len(label_vocab), bos_tag_id=label_vocab.to_indices(label_vocab.bos_token),
                        eos_tag_id=label_vocab.to_indices(label_vocab.eos_token),
                        pad_tag_id=label_vocab.to_indices(label_vocab.padding_token))

    def forward(self, x: torch.Tensor) -> Tuple[torch.tensor, torch.tensor]:
        masking = x.ne(self._embedding._padding_idx).float()
        fmap = self._embedding(x)
        fmap = self._pipe(fmap)
        hiddens = self._bilstm(fmap)
        emissions = self._fc(hiddens)
        score, path = self._crf.decode(emissions, mask=masking)
        return score, path

    def loss(self, x: torch.Tensor, y:torch.Tensor) -> torch.Tensor:
        masking = x.ne(self._embedding._padding_idx).float()
        fmap = self._embedding(x)
        fmap = self._pipe(fmap)
        hiddens = self._bilstm(fmap)
        emissions = self._fc(hiddens)
        nll = self._crf(emissions, y, mask=masking)
        return nll
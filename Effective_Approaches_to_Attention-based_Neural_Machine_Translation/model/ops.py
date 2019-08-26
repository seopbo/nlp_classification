import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
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
dl = DataLoader(ds, 2, shuffle=False, num_workers=4, collate_fn=batchify)
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

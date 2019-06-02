import pandas as pd
import torch
from torch.utils.data import Dataset
from gluonnlp import Vocab
from typing import Tuple


class Corpus(Dataset):
    """Corpus class"""
    def __init__(self, filepath: str, token_vocab: Vocab, label_vocab: Vocab) -> None:
        """Instantiating Corpus class

        Args:
            filepath (str): filepath
            token_vocab: (gluonnlp.Vocab): the instance of gluonnlp.Vocab that has token information
            label_vocab (gluonnlp.Vocab): the instance of gluonnlp.Vocab that has label information
        """
        self._corpus = pd.read_pickle(filepath)
        self._token_vocab = token_vocab
        self._label_vocab = label_vocab

    def __len__(self) -> int:
        return len(self._corpus)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens, labels = map(lambda elm: elm, self._corpus[idx])
        tokens2indices = torch.tensor([self._token_vocab.to_indices(token) for token in tokens])
        labels2indices = torch.tensor([self._label_vocab.to_indices(label) for label in labels])
        length = torch.tensor(len(tokens2indices))
        return tokens2indices, labels2indices, length


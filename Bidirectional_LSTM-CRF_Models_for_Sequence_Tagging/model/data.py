import pandas as pd
import torch
from torch.utils.data import Dataset
from gluonnlp import Vocab
from typing import Tuple, List, Callable


class Corpus(Dataset):
    """Corpus class"""
    def __init__(self, filepath: str, token_transform_fn: Callable[[str], List[int]],
                 label_transform_fn: Callable[[str], List[int]]) -> None:
        """Instantiating Corpus class

        Args:
            filepath (str): filepath
            token_transform_fn (Callable): a function that can act as a transformer for token
            label_transform_fn (Callable): a function that can act as a transformer for label
        """
        self._corpus = pd.read_pickle(filepath)
        self._token_transform = token_transform_fn
        self._label_transform = label_transform_fn

    def __len__(self) -> int:
        return len(self._corpus)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens, labels = map(lambda elm: elm, self._corpus[idx])
        tokens2indices = torch.tensor(self._token_transform(tokens))
        labels2indices = torch.tensor(self._label_transform(labels))
        length = torch.tensor(len(tokens2indices))
        return tokens2indices, labels2indices, length


class Tokenizer:
    """Tokenizer class"""
    def __init__(self, vocab: Vocab, split_fn: Callable[[str], List[str]],
                 pad_fn: Callable[[List[int]], List[int]] = None) -> None:
        """Instantiating Tokenizer class
        Args:
            vocab (gluonnlp.data.Vocab): the instance of gluonnlp.Vocab created from specific split_fn
            split_fn (Callable): a function that can act as a splitter
            pad_fn (Callable): a function that can act as a padder
        """
        self._vocab = vocab
        self._split = split_fn
        self._pad = pad_fn

    def split(self, string: str) -> List[str]:
        list_of_tokens = self._split(string)
        return list_of_tokens

    def transform(self, list_of_tokens: List[str]) -> List[int]:
        list_of_indices = self._vocab.to_indices(list_of_tokens)
        list_of_indices = self._pad(list_of_indices) if self._pad else list_of_indices
        return list_of_indices

    def split_and_transform(self, string: str) -> List[int]:
        return self.transform(self.split(string))

    @property
    def vocab(self):
        return self._vocab
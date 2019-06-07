import pandas as pd
import torch
from gluonnlp import Vocab
from torch.utils.data import Dataset
from typing import Tuple, Callable, List


class Corpus(Dataset):
    """Corpus class"""
    def __init__(self, filepath: str, transform_fn: Callable[[str], List[int]],
                 min_length: int, pad_val: int) -> None:
        """Instantiating Corpus class

        Args:
            filepath (str): filepath
            transform_fn (Callable): a function that can act as a transformer
            min_length (int): minimum length of sequence
                             (if length is lower than min_length, the sequence is padded to min_length)
            pad_val (int): the index of padding
        """
        self._corpus = pd.read_csv(filepath, sep='\t').loc[:, ['document', 'label']]
        self._transform = transform_fn
        self._min_length = min_length
        self._pad_val = pad_val

    def __len__(self) -> int:
        return len(self._corpus)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens2indices = self._transform(self._corpus.iloc[idx]['document'])
        if len(tokens2indices) < self._min_length:
            tokens2indices = tokens2indices + (self._min_length - len(tokens2indices)) * [self._pad_val]
        tokens2indices = torch.tensor(tokens2indices)
        label = torch.tensor(self._corpus.iloc[idx]['label'])
        length = torch.tensor(len(tokens2indices))
        return tokens2indices, label, length


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
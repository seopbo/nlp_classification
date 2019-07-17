import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Tuple, Callable, List
from torch.nn.utils.rnn import pad_sequence


class Corpus(Dataset):
    """Corpus class"""
    def __init__(self, filepath: str, transform_fn: Callable[[str], List[int]],
                 min_length: int, pad_val: int = 1) -> None:
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
        return tokens2indices, label


def batchify(data: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """custom collate_fn for DataLoader

    Args:
        data (list): list of torch.Tensors

    Returns:
        data (tuple): tuple of torch.Tensors
    """
    indices, labels = zip(*data)
    indices = pad_sequence(indices, batch_first=True, padding_value=1)
    labels = torch.stack(labels, 0)
    return indices, labels

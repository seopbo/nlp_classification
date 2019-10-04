import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
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
        return tokens2indices, labels2indices


def batchify(data: List[Tuple[torch.tensor, torch.tensor, torch.tensor]]) ->\
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """custom collate_fn for DataLoader

    Args:
        data (list): list of torch.Tensors

    Returns:
        data (tuple): tuple of torch.Tensors
    """
    tokens2indices, labels2indices = zip(*data)
    tokens2indices = pad_sequence(tokens2indices, batch_first=True, padding_value=1)
    labels2indices = pad_sequence(labels2indices, batch_first=True, padding_value=0)
    return tokens2indices, labels2indices

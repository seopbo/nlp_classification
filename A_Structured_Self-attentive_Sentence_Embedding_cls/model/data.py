import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Tuple, List, Callable


class Corpus(Dataset):
    """Corpus class"""
    def __init__(self, filepath: str, transform_fn: Callable[[str], List[int]]) -> None:
        """Instantiating Corpus class
        Args:
            filepath (str): filepath
            transform_fn (Callable): a function that can act as a transformer
        """
        self._corpus = pd.read_csv(filepath, sep='\t').loc[:, ['document', 'label']]
        self._transform = transform_fn

    def __len__(self) -> int:
        return len(self._corpus)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens2indices = torch.tensor(self._transform(self._corpus.iloc[idx]['document']))
        label = torch.tensor(self._corpus.iloc[idx]['label'])
        return tokens2indices, label


def batchify(data: List[Tuple[torch.tensor, torch.tensor, torch.tensor]]) ->\
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """custom collate_fn for DataLoader

    Args:
        data (list): list of torch.Tensors

    Returns:
        data (tuple): tuple of torch.Tensors
    """
    indices, labels = zip(*data)
    indices = pad_sequence(indices, batch_first=True, padding_value=1, )
    labels = torch.stack(labels, 0)
    return indices, labels

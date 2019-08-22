import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Tuple, List, Callable


class Corpus(Dataset):
    """Corpus class"""
    def __init__(self, filepath: str, transform_fn: Callable[[str], List[int]]) -> None:
        """Instantiating Corpus class

        Args:
            filepath (str): filepath
            transform_fn (Callable): a function that can act as a transformer
        """
        self._corpus = pd.read_csv(filepath, sep='\t')
        self._transform = transform_fn

    def __len__(self) -> int:
        return len(self._corpus)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        q1, q2, is_duplicate = self._corpus.iloc[idx].tolist()
        list_of_indices, list_of_token_types = [torch.tensor(elm) for elm in self._transform(q1, q2)]
        label = torch.tensor(is_duplicate)
        return list_of_indices, list_of_token_types, label

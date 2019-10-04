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
        self._corpus = pd.read_csv(filepath, sep="\t")
        self._transform = transform_fn

    def __len__(self) -> int:
        return len(self._corpus)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        qa, qb, is_duplicate = self._corpus.iloc[idx].tolist()
        list_of_qa = torch.tensor(self._transform(qa))
        list_of_qb = torch.tensor(self._transform(qb))
        label = torch.tensor(is_duplicate)
        return list_of_qa, list_of_qb, label


def batchify(
    data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """custom collate_fn for DataLoader

    Args:
        data (List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]): list of tuples of torch.Tensors

    Returns:
        data (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): tuple of torch.Tensors

    """
    data = list(zip(*data))
    queries_a, queries_b, is_duplicates = data
    queries_a = pad_sequence(queries_a, batch_first=True, padding_value=1)
    queries_b = pad_sequence(queries_b, batch_first=True, padding_value=1)
    is_duplicates = torch.stack(is_duplicates, 0)

    return queries_a, queries_b, is_duplicates

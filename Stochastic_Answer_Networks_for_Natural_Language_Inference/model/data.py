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
        qa_coarse, qa_fine = [torch.tensor(elm) for elm in self._transform(qa)]
        qb_coarse, qb_fine = [torch.tensor(elm) for elm in self._transform(qb)]
        label = torch.tensor(is_duplicate)
        return (qa_coarse, qa_fine), (qb_coarse, qb_fine), label


def batchify(
    data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

    qa, qb, label = zip(*data)
    qa_coarse, qa_fine = zip(*qa)
    qb_coarse, qb_fine = zip(*qb)

    qa_coarse = pad_sequence(qa_coarse, batch_first=True, padding_value=1)
    qa_fine = pad_sequence(qa_fine, batch_first=False, padding_value=1).permute(1, 0, 2)

    qb_coarse = pad_sequence(qb_coarse, batch_first=True, padding_value=1)
    qb_fine = pad_sequence(qb_fine, batch_first=False, padding_value=1).permute(1, 0, 2)

    label = torch.stack(label, 0)

    return (qa_coarse, qa_fine), (qb_coarse, qb_fine), label

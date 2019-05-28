import pandas as pd
import torch
from torch.utils.data import Dataset
from gluonnlp import Vocab
from typing import Tuple, List, Callable


class Corpus(Dataset):
    """Corpus class"""
    def __init__(self, filepath: str, tokenizer: Callable[[str], List[str]], vocab: Vocab) -> None:
        """Instantiating Corpus class

        Args:
            filepath (str): filepath
            tokenizer (Callable): a function that can act as a tokenizer
            vocab (gluonnlp.Vocab): the instance of gluonnlp.Vocab
        """
        self._corpus = pd.read_csv(filepath, sep='\t').loc[:, ['document', 'label']]
        self._tokenizer = tokenizer
        self._vocab = vocab

    def __len__(self) -> int:
        return len(self._corpus)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokenized = self._tokenizer(self._corpus.iloc[idx]['document'])
        tokenized2indices = torch.tensor([self._vocab.to_indices(token) for token in tokenized])
        label = torch.tensor(self._corpus.iloc[idx]['label'])
        length = torch.tensor(len(tokenized2indices))
        return tokenized2indices, label, length
import pandas as pd
import torch
from torch.utils.data import Dataset
from model.utils import JamoTokenizer
from typing import Tuple


class Corpus(Dataset):
    """Corpus class"""
    def __init__(self, filepath: str, tokenizer: JamoTokenizer, min_length: int) -> None:
        """Instantiating Corpus class

        Args:
            filepath (str): filepath
            tokenizer (JamoTokenizer): instance of JamoTokenizer
            min_length (int): minimum length of query
                             (if length is lower than min_length, the sequences is padded to min_length)
        """
        self._corpus = pd.read_csv(filepath, sep='\t').loc[:, ['document', 'label']]
        self._tokenizer = tokenizer
        self._min_length = min_length

    def __len__(self) -> int:
        return len(self._corpus)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokenized2indices = self._tokenizer.tokenize_and_transform(self._corpus.iloc[idx]['document'])
        if len(tokenized2indices) < self._min_length:
            tokenized2indices = tokenized2indices + (self._min_length - len(tokenized2indices)) * \
                                [self._tokenizer.token2idx.get(' ')]
        tokenized2indices = torch.tensor(tokenized2indices)
        length = torch.tensor(len(tokenized2indices))
        label = torch.tensor(self._corpus.iloc[idx]['label'])
        return tokenized2indices, label, length
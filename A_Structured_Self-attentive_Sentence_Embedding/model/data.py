import pandas as pd
import torch
from torch.utils.data import Dataset
from mecab import MeCab
from gluonnlp import Vocab
from typing import Tuple


class Corpus(Dataset):
    """Corpus class"""
    def __init__(self, filepath: str, tokenizer: MeCab, vocab: Vocab) -> None:
        self._corpus = pd.read_csv(filepath, sep='\t').loc[:, ['document', 'label']]
        self._tokenizer = tokenizer
        self._vocab = vocab

    def __len__(self) -> int:
        return len(self._corpus)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokenized = self._tokenizer.morphs(self._corpus.iloc[idx]['document'])
        tokenized2indices = torch.tensor([self._vocab.token_to_idx[token] for token in tokenized])
        label = torch.tensor(self._corpus.iloc[idx]['label'])
        length = torch.tensor(len(tokenized2indices))
        return tokenized2indices, label, length
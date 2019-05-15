import pandas as pd
import torch
from torch.utils.data import Dataset
from mecab import MeCab
from gluonnlp import Vocab
from typing import Tuple


class Corpus(Dataset):
    """Corpus class"""
    def __init__(self, filepath: str, tokenizer: MeCab, vocab: Vocab) -> None:
        """Instantiating Corpus class
        Args:
            filepath (str): filepath
            tokenizer (mecab.MeCab): the instance of mecab.Mecab
            padder (gluonnlp.data.PadSequence): the instance of gluonnlp.data.PadSequence
            vocab (gluonnlp.Vocab): the instance of gluonnlp.Vocab
        """
        self._corpus = pd.read_csv(filepath)
        self._tokenizer = tokenizer
        self._vocab = vocab

    def __len__(self) -> int:
        return len(self._corpus)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_a = self._tokenizer.morphs(self._corpus.iloc[idx]['question1'])
        query_b = self._tokenizer.morphs(self._corpus.iloc[idx]['question2'])
        query_a_indices = torch.tensor([self._vocab.token_to_idx[token] for token in query_a])
        query_b_indices = torch.tensor([self._vocab.token_to_idx[token] for token in query_b])
        is_duplicate = torch.tensor(self._corpus.iloc[idx]['is_duplicate'])
        query_a_length = torch.tensor(len(query_a_indices))
        query_b_length = torch.tensor(len(query_b_indices))
        return (query_a_indices, query_a_length), (query_b_indices, query_b_length), is_duplicate
import pandas as pd
import torch
from torch.utils.data import Dataset
from mecab import MeCab
from gluonnlp.data import PadSequence
from gluonnlp import Vocab
from typing import Tuple

class Corpus(Dataset):
    """Corpus class"""
    def __init__(self, filepath: str, vocab: Vocab, tokenizer: MeCab, padder: PadSequence) -> None:
        """Instantiating Corpus class

        Args:
            filepath (str): filepath
            vocab (gluonnlp.Vocab): instance of gluonnlp.Vocab
            tokenizer (mecab.Mecab): instance of mecab.Mecab
            padder (gluonnlp.data.PadSequence): instance of gluonnlp.data.PadSequence
        """
        self._corpus = pd.read_csv(filepath, sep='\t').loc[:, ['document', 'label']]
        self._vocab = vocab
        self._toknizer = tokenizer
        self._padder = padder

    def __len__(self) -> int:
        return len(self._corpus)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tokenized = self._toknizer.morphs(self._corpus.iloc[idx]['document'])
        tokenized2indices = torch.tensor(self._padder([self._vocab.token_to_idx[token] for token in tokenized]))
        label = torch.tensor(self._corpus.iloc[idx]['label'])
        return tokenized2indices, label


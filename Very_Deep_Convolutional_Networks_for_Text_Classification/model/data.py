import pandas as pd
import torch
from torch.utils.data import Dataset
from gluonnlp.data import PadSequence
from model.utils import JamoTokenizer

class Corpus(Dataset):
    """Corpus class"""
    def __init__(self, filepath: str, tokenizer: JamoTokenizer, padder: PadSequence) -> None:
        """Instantiating Corpus class

        Args:
            filepath (str): filepath
            tokenizer (JamoTokenizer): instance of model.utils.JamoTokenizer
            padder (gluonnlp.data.PadSequence): instance of gluonnlp.data.PadSequence
        """
        self._corpus = pd.read_table(filepath).loc[:, ['document', 'label']]
        self._padder = padder
        self._tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self._corpus)

    def __getitem__(self, idx) -> (torch.Tensor, torch.Tensor):
        tokenized2indices = self._tokenizer.tokenize_and_transform(self._corpus.iloc[idx]['document'])
        tokenized2indices = torch.tensor(self._padder(tokenized2indices))
        label = torch.tensor(self._corpus.iloc[idx]['label'])
        return tokenized2indices, label

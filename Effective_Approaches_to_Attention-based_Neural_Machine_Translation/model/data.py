import pandas as pd
from torch.utils.data import Dataset


class ParCorpus(Dataset):
    def __init__(self, filepath, src_transform_fn, tgt_transform_fn):
        self._corpus = pd.read_csv(filepath, sep='\t', index=False)
        self._src_transform = src_transform_fn
        self._src_transform = tgt_transform_fn
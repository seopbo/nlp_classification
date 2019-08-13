import pandas as pd
import itertools
import pickle
from collections import Counter
from pathlib import Path
from model.split import split_morphs, split_space
from model.utils import Vocab

data_dir = Path('data')
list_of_dataset = sorted(data_dir.iterdir())
tr_dataset = pd.read_csv(list_of_dataset[-1])

# korean vocab
count_ko = Counter(itertools.chain.from_iterable(tr_dataset['ko'].apply(split_morphs).tolist()))
list_of_token_ko = [token[0] for token in count_ko.items() if token[1] >= 10]
vocab_ko = Vocab(list_of_token_ko, bos_token=None, eos_token=None)

with open(data_dir / 'vocab_ko.pkl', mode='wb') as io:
    pickle.dump(vocab_ko, io)

# english vocab
count_en = Counter(itertools.chain.from_iterable(tr_dataset['en'].apply(split_space).tolist()))
list_of_token_en = [token[0] for token in count_en.items() if token[1] >= 10]
vocab_en = Vocab(list_of_token_en)

with open(data_dir / 'vocab_en.pkl', mode='wb') as io:
    pickle.dump(vocab_en, io)

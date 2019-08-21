import pickle
import pandas as pd
import itertools
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from mecab import MeCab
from model.utils import Vocab

# loading dataset
cwd = Path.cwd()
dataset = pd.read_csv(cwd / 'data' / 'kor_pair_train.csv').filter(items=['question1', 'question2', 'is_duplicate'])
tst = pd.read_csv(cwd / 'data' / 'kor_pair_test.csv').filter(items=['question1', 'question2', 'is_duplicate'])
total = pd.concat([dataset, tst], axis=0, ignore_index=True, sort=False)
tr, val = train_test_split(total, test_size=.2, random_state=777)

# extracting morph in sentences
list_of_tokens = pd.concat([tr['question1'], tr['question2']]).apply(MeCab().morphs).tolist()

# making the vocab
min_freq = 5
count_tokens = Counter(itertools.chain.from_iterable(list_of_tokens))
list_of_tokens = [count_token[0] for count_token in count_tokens.items() if count_token[1] >= min_freq]
vocab = Vocab(list_of_tokens, bos_token=None, eos_token=None)

# saving vocab
with open(cwd / 'data' / 'vocab.pkl', mode='wb') as io:
    pickle.dump(vocab, io)

tr.to_csv(cwd / 'data' / 'train.txt', index=False, sep='\t')
val.to_csv(cwd / 'data' / 'val.txt', index=False, sep='\t')

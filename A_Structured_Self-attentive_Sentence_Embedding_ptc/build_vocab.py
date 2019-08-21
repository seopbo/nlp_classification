import pickle
import pandas as pd
import gluonnlp as nlp
import itertools
from pathlib import Path
from sklearn.model_selection import train_test_split
from mecab import MeCab

# loading dataset
cwd = Path.cwd()
dataset = pd.read_csv(cwd / 'data' / 'kor_pair_train.csv').filter(items=['question1', 'question2', 'is_duplicate'])
tst = pd.read_csv(cwd / 'data' / 'kor_pair_test.csv').filter(items=['question1', 'question2', 'is_duplicate'])
total = pd.concat([dataset, tst], axis=0, ignore_index=True, sort=False)
tr, val = train_test_split(total, test_size=.2, random_state=777)

# extracting morph in sentences
list_of_tokens = pd.concat([tr['question1'], tr['question2']]).apply(MeCab().morphs).tolist()

# making the vocab
counter = nlp.data.count_tokens(itertools.chain.from_iterable(list_of_tokens))
vocab = nlp.Vocab(counter=counter, min_freq=5, bos_token=None, eos_token=None)

# # connecting SISG embedding with vocab
ptr_embedding = nlp.embedding.create('fasttext', source='wiki.ko')
vocab.set_embedding(ptr_embedding)

# saving vocab
with open(cwd / 'data' / 'vocab.pkl', mode='wb') as io:
    pickle.dump(vocab, io)


tr.to_csv(cwd / 'data' / 'train.txt', index=False)
val.to_csv(cwd / 'data' / 'val.txt', index=False)
import pickle
import pandas as pd
import gluonnlp as nlp
import itertools
from pathlib import Path
from sklearn.model_selection import train_test_split
from mecab import MeCab

# loading dataset
proj_dir = Path.cwd()
dataset = pd.read_csv(proj_dir / 'data' / 'kor_pair_train.csv').filter(items=['question1', 'question2',
                                                                              'is_duplicate'])
tst = pd.read_csv(proj_dir / 'data' / 'kor_pair_test.csv').filter(items=['question1', 'question2',
                                                                         'is_duplicate'])


# extracting morph in sentences
tokenizer = MeCab()
tokenized = pd.concat([dataset['question1'], dataset['question2']]).apply(tokenizer.morphs).tolist()

# making the vocab
counter = nlp.data.count_tokens(itertools.chain.from_iterable(tokenized))
vocab = nlp.Vocab(counter=counter, min_freq=10, bos_token=None, eos_token=None)

# connecting SISG embedding with vocab
ptr_embedding = nlp.embedding.create('fasttext', source='wiki.ko', load_ngrams=True)
vocab.set_embedding(ptr_embedding)

# saving vocab
with open(proj_dir / 'data' / 'vocab.pkl', mode='wb') as io:
    pickle.dump(vocab, io)


tr, val = train_test_split(dataset, test_size=.1, random_state=777)
tr.to_csv(proj_dir / 'data' / 'train.txt', index=False)
val.to_csv(proj_dir / 'data' / 'val.txt', index=False)
tst.to_csv(proj_dir / 'data' / 'test.txt', index=False)
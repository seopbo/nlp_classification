import pandas as pd
import gluonnlp as nlp
import itertools
import pickle
from pathlib import Path
from mecab import MeCab
from sklearn.model_selection import train_test_split

# loading dataset
proj_dir = Path('.')
tr_filepath = proj_dir / 'data/ratings_train.txt'
data = pd.read_csv(tr_filepath, sep='\t').loc[:, ['document', 'label']]
data = data.loc[data['document'].isna().apply(lambda elm : not elm), :]

tr_data, val_data = train_test_split(data, test_size=.2)

tst_filepath = proj_dir / 'data/ratings_test.txt'
tst_data = pd.read_csv(tst_filepath, sep='\t').loc[:, ['document', 'label']]
tst_data = tst_data.loc[tst_data['document'].isna().apply(lambda elm : not elm), :]

# extracting morph in sentences
tokenizer = MeCab()
tokenized = tr_data['document'].apply(tokenizer.morphs).tolist()

# making the vocab
counter = nlp.data.count_tokens(itertools.chain.from_iterable(tokenized))
vocab = nlp.Vocab(counter=counter, min_freq=10, bos_token=None, eos_token=None)

# connecting SISG embedding with vocab
ptr_embedding = nlp.embedding.create('fasttext', source='wiki.ko')
vocab.set_embedding(ptr_embedding)

# saving vocab
with open('./data/vocab.pkl', mode='wb') as io:
    pickle.dump(vocab, io)

# saving tr_data, val_data, tst_data
tr_data.to_csv('./data/train.txt', index=False, sep='\t')
val_data.to_csv('./data/val.txt', index=False, sep='\t')
tst_data.to_csv('./data/test.txt', index=False, sep='\t')

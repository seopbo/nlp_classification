import pandas as pd
import itertools
import pickle
import gluonnlp as nlp
from collections import Counter
from pathlib import Path
from model.split import split_morphs, split_space
from model.utils import Vocab

data_dir = Path('data')
tr_filepath = (data_dir / 'train').with_suffix('.txt')
tr_dataset = pd.read_csv(tr_filepath, sep='\t')

# korean vocab
count_ko = Counter(itertools.chain.from_iterable(tr_dataset['ko'].apply(split_morphs).tolist()))
list_of_token_ko = sorted([token[0] for token in count_ko.items() if token[1] >= 20])
tmp_vocab = nlp.Vocab(Counter(list_of_token_ko), bos_token=None, eos_token=None)
ptr_embedding = nlp.embedding.create('fasttext', source='wiki.ko', load_ngrams=True)
tmp_vocab.set_embedding(ptr_embedding)
array = tmp_vocab.embedding.idx_to_vec.asnumpy()


vocab_ko = Vocab(list_of_token_ko, bos_token=None, eos_token=None)
vocab_ko.embedding = array

with open(data_dir / 'vocab_ko.pkl', mode='wb') as io:
    pickle.dump(vocab_ko, io)

# english vocab
count_en = Counter(itertools.chain.from_iterable(tr_dataset['en'].apply(split_space).tolist()))
list_of_token_en = [token[0] for token in count_en.items() if token[1] >= 20]
tmp_vocab = nlp.Vocab(Counter(list_of_token_en))
ptr_embedding = nlp.embedding.create('fasttext', source='wiki.simple', load_ngrams=True)
tmp_vocab.set_embedding(ptr_embedding)
array = tmp_vocab.embedding.idx_to_vec.asnumpy()
vocab_en = Vocab(list_of_token_en)
vocab_en.embedding = array

with open(data_dir / 'vocab_en.pkl', mode='wb') as io:
    pickle.dump(vocab_en, io)

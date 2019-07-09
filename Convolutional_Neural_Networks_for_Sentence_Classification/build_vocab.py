import itertools
import pickle
import gluonnlp as nlp
import pandas as pd
from mecab import MeCab
from model.utils import Vocab
from utils import Config
from collections import Counter

# loading dataset
data_config = Config('data/config.json')
tr = pd.read_csv(data_config.train, sep='\t').loc[:, ['document', 'label']]

# extracting morph in sentences
split_fn = MeCab().morphs
list_of_tokens = tr['document'].apply(split_fn).tolist()

# generating the vocab
min_freq = 10
token_counter = Counter(itertools.chain.from_iterable(list_of_tokens))
list_of_tokens = [token_count[0] for token_count in token_counter.items() if token_count[1] >= min_freq]
list_of_tokens = sorted(list_of_tokens)
list_of_tokens.insert(0, '<pad>')
list_of_tokens.insert(0, '<unk>')

tmp_vocab = nlp.Vocab(counter=Counter(list_of_tokens), min_freq=1, bos_token=None, eos_token=None)

# connecting SISG embedding with vocab
ptr_embedding = nlp.embedding.create('fasttext', source='wiki.ko')
tmp_vocab.set_embedding(ptr_embedding)
array = tmp_vocab.embedding.idx_to_vec.asnumpy()

vocab = Vocab(list_of_tokens, padding_token='<pad>', unknown_token='<unk>', bos_token=None, eos_token=None)
vocab.embedding = array

# saving vocab
with open('data/vocab.pkl', mode='wb') as io:
    pickle.dump(vocab, io)
data_config.vocab = 'data/vocab.pkl'
data_config.save('data/config.json')

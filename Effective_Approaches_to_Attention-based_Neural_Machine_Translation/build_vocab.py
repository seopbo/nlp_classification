import pandas as pd
import itertools
import pickle
import gluonnlp as nlp
from collections import Counter
from pathlib import Path
from model.split import Stemmer
from model.utils import Vocab
from utils import Config

# loading dataset
sample_dir = Path('sample')
config = Config("conf/dataset/sample.json")
tr = pd.read_csv(config.train, sep='\t')

# korean vocab
split_ko = Stemmer(language='ko')
count_ko = Counter(itertools.chain.from_iterable(tr['ko'].apply(split_ko.extract_stem).tolist()))
tmp_vocab = nlp.Vocab(count_ko, bos_token=None, eos_token=None)
ptr_embedding = nlp.embedding.create('fasttext', source='wiki.ko', load_ngrams=True)
tmp_vocab.set_embedding(ptr_embedding)
array = tmp_vocab.embedding.idx_to_vec.asnumpy()

vocab_ko = Vocab(tmp_vocab.idx_to_token, bos_token=None, eos_token=None)
vocab_ko.embedding = array
vocab_ko_filepath = sample_dir / "vocab_ko.pkl"
config.update({"source_vocab": str(vocab_ko_filepath)})

with open(vocab_ko_filepath, mode='wb') as io:
    pickle.dump(vocab_ko, io)

# english vocab
split_en = Stemmer(language='en')
count_en = Counter(itertools.chain.from_iterable(tr['en'].apply(split_en.extract_stem).tolist()))
tmp_vocab = nlp.Vocab(count_en)
ptr_embedding = nlp.embedding.create('fasttext', source='wiki.simple', load_ngrams=True)
tmp_vocab.set_embedding(ptr_embedding)
array = tmp_vocab.embedding.idx_to_vec.asnumpy()

vocab_en = Vocab(tmp_vocab.idx_to_token)
vocab_en.embedding = array
vocab_en_filepath = sample_dir / "vocab_en.pkl"
config.update({"target_vocab": str(vocab_en_filepath)})

with open(vocab_en_filepath, mode='wb') as io:
    pickle.dump(vocab_en, io)

config.save("conf/dataset/sample.json")

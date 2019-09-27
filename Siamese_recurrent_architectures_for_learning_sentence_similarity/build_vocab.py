import pickle
import pandas as pd
import itertools
import gluonnlp as nlp
from pathlib import Path
from collections import Counter
from model.split import split_morphs
from model.utils import Vocab

cwd = Path.cwd()
data_dir = cwd / "data"
train = pd.read_csv(data_dir / "train.txt", sep="\t")

list_of_tokens_qa = train["question1"].apply(lambda sen: split_morphs(sen)).tolist()
list_of_tokens_qb = train["question2"].apply(lambda sen: split_morphs(sen)).tolist()
list_of_tokens = list_of_tokens_qa + list_of_tokens_qb

count_tokens = Counter(itertools.chain.from_iterable(list_of_tokens))
tmp_vocab = nlp.Vocab(counter=count_tokens, bos_token=None, eos_token=None)
ptr_embedding = nlp.embedding.create("fasttext", source="wiki.ko", load_ngrams=True)
tmp_vocab.set_embedding(ptr_embedding)

vocab = Vocab(tmp_vocab.idx_to_token, bos_token=None, eos_token=None)
vocab.embedding = tmp_vocab.embedding.idx_to_vec.asnumpy()

with open(data_dir / "vocab.pkl", mode="wb") as io:
    pickle.dump(vocab, io)

import pickle
import pandas as pd
import itertools
import gluonnlp as nlp
from pathlib import Path
from collections import Counter
from model.split import split_morphs
from model.utils import Vocab
from utils import Config

# morphs
data_dir = Path("data")
config = Config(data_dir / "config.json")
train = pd.read_csv(config.train, sep="\t")

list_of_tokens_qa = train["question1"].apply(lambda sen: split_morphs(sen)).tolist()
list_of_tokens_qb = train["question2"].apply(lambda sen: split_morphs(sen)).tolist()
list_of_tokens = list_of_tokens_qa + list_of_tokens_qb

count_tokens = Counter(itertools.chain.from_iterable(list_of_tokens))
tmp_vocab = nlp.Vocab(counter=count_tokens, bos_token=None, eos_token=None)
ptr_embedding = nlp.embedding.create("fasttext", source="wiki.ko", load_ngrams=True)
tmp_vocab.set_embedding(ptr_embedding)

morph_vocab = Vocab(tmp_vocab.idx_to_token, bos_token=None, eos_token=None)
morph_vocab.embedding = tmp_vocab.embedding.idx_to_vec.asnumpy()

with open(data_dir / "morph_vocab.pkl", mode="wb") as io:
    pickle.dump(morph_vocab, io)

config.update({"coarse_vocab": str(data_dir / "morph_vocab.pkl")})

# jamo
chosung_list = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ',
                'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ',
                'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

jungsung_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ',
                 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ',
                 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']

jongsung_list = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ',
                 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ',
                 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

list_of_jamos = sorted(set(chosung_list + jungsung_list + jongsung_list))
jamo_vocab = Vocab(list_of_tokens=list_of_jamos, bos_token=None, eos_token=None)

with open(data_dir / 'jamo_vocab.pkl', mode='wb') as io:
    pickle.dump(jamo_vocab, io)

config.update({"fine_vocab": str(data_dir / "jamo_vocab.pkl")})
config.save(data_dir / "config.json")

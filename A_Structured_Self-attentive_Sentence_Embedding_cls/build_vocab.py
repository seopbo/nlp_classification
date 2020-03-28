import itertools
import pickle
import gluonnlp as nlp
import pandas as pd
from pathlib import Path
from model.utils import Vocab
from model.split import split_morphs
from utils import Config
from collections import Counter

# loading dataset
nsmc_dir = Path("nsmc")
config = Config("conf/dataset/nsmc.json")
tr = pd.read_csv(config.train, sep="\t").loc[:, ["document", "label"]]

# extracting morph in sentences
list_of_tokens = tr["document"].apply(split_morphs).tolist()

# generating the vocab
token_counter = Counter(itertools.chain.from_iterable(list_of_tokens))
tmp_vocab = nlp.Vocab(
    counter=token_counter, min_freq=10, bos_token=None, eos_token=None
)

# connecting SISG embedding with vocab
ptr_embedding = nlp.embedding.create("fasttext", source="wiki.ko")
tmp_vocab.set_embedding(ptr_embedding)
array = tmp_vocab.embedding.idx_to_vec.asnumpy()

vocab = Vocab(
    tmp_vocab.idx_to_token,
    padding_token="<pad>",
    unknown_token="<unk>",
    bos_token=None,
    eos_token=None,
)
vocab.embedding = array

# saving vocab
with open(nsmc_dir / "vocab.pkl", mode="wb") as io:
    pickle.dump(vocab, io)

config.update({"vocab": str(nsmc_dir / "vocab.pkl")})
config.save("conf/dataset/nsmc.json")

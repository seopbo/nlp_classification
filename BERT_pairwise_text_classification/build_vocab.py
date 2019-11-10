import pickle
from pathlib import Path
from model.utils import Vocab
from utils import Config
from pretrained.tokenization import BertTokenizer

# loading BertTokenizer
ptr_tokenizer = BertTokenizer.from_pretrained('pretrained/vocab.korean.rawtext.list', do_lower_case=False)
idx_to_token = list(ptr_tokenizer.vocab.keys())
token_to_idx = {token: idx for idx, token in enumerate(idx_to_token)}

# generate vocab
vocab = Vocab(
    idx_to_token,
    padding_token="[PAD]",
    unknown_token="[UNK]",
    bos_token=None,
    eos_token=None,
    reserved_tokens=["[CLS]", "[SEP]", "[MASK]"],
    token_to_idx=token_to_idx,
)

# save vocab
data_dir = Path("data")
with open(data_dir / "vocab.pkl", mode="wb") as io:
    pickle.dump(vocab, io)

config = Config(data_dir / "config.json")
config.update({"vocab": str(data_dir / "vocab.pkl")})
config.save(data_dir / "config.json")

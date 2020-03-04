import pickle
from pathlib import Path
from model.utils import Vocab
from utils import Config

CHOSUHG_LIST = ["ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
                "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
JUNGSUNG_LIST = ["ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ",
                 "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ"]
JONGSUNG_LIST = [" ", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ",
                 "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ",
                 "ㅋ", "ㅌ", "ㅍ", "ㅎ"]

list_of_jamos = sorted(set(CHOSUHG_LIST + JUNGSUNG_LIST + JONGSUNG_LIST))
vocab = Vocab(list_of_tokens=list_of_jamos, bos_token=None, eos_token=None)

nsmc_dir = Path("nsmc")
with open(nsmc_dir / "vocab.pkl", mode="wb") as io:
    pickle.dump(vocab, io)

config = Config("conf/dataset/nsmc.json")
config.update({"vocab": str(nsmc_dir / "vocab.pkl")})
config.save("conf/dataset/nsmc.json")

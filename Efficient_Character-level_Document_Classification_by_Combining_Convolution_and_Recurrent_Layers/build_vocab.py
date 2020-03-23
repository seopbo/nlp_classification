import pickle
from pathlib import Path
from model.utils import Vocab
from utils import Config

LIST_OF_CHOSUNG = [
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ",
    "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ",
    "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
LIST_OF_JUNGSUNG = [
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ",
    "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ",
    "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ",
    "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ",
    "ㅣ"]
LIST_OF_JONGSUNG = [
    " ", "ㄱ", "ㄲ", "ㄳ", "ㄴ",
    "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ",
    "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ",
    "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ",
    "ㅌ", "ㅍ", "ㅎ"]

LIST_OF_JAMOS = sorted(set(LIST_OF_CHOSUNG + LIST_OF_JUNGSUNG + LIST_OF_JONGSUNG))
vocab = Vocab(list_of_tokens=LIST_OF_JAMOS, bos_token=None, eos_token=None)
nsmc_dir = Path("nsmc")

with open(nsmc_dir / "vocab.pkl", mode="wb") as io:
    pickle.dump(vocab, io)

config = Config("conf/dataset/nsmc.json")
config.update({"vocab": str(nsmc_dir / "vocab.pkl")})
config.save("conf/dataset/nsmc.json")

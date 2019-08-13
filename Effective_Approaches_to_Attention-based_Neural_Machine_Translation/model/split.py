import re
import string
import unicodedata
from mecab import MeCab
from typing import List

split_morphs = MeCab().morphs


def split_space(sentence: str) -> List[str]:
    return re.split(r'\s+', sentence)


def preprocess(sentence):
    sentence = re.sub(r'[' + re.escape(string.punctuation) + ']', '', sentence)
    sentence = re.sub(r'\s{2,}', ' ', sentence)
    sentence = unicodedata.normalize('NFKD', sentence)
    sentence = sentence.strip()
    sentence = sentence.lower()
    return sentence

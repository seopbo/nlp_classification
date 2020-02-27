import re
import unicodedata
from konlpy.tag import Mecab
from stemming.porter2 import stem


class Stemmer:
    def __init__(self, language):
        punct = '"“”#$%&\'()*+,-/:;<=>@[\\]^_`{|}~'
        self._table = str.maketrans({key: None for key in punct})
        self._morphs = Mecab().morphs
        self._language = language

    @staticmethod
    def unicode_to_ascii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    # Lowercase, trim, and remove non-letter characters
    @staticmethod
    def normalize_string(s):
        s = Stemmer.unicode_to_ascii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    def extract_stem(self, sentence):
        if self._language == 'ko':
            spaced = self._morphs(unicodedata.normalize('NFKC', sentence.strip()).translate(self._table))
        elif self._language == 'en':
            spaced = [stem(j) for j in self._morphs(self.normalize_string(sentence))]
        return spaced

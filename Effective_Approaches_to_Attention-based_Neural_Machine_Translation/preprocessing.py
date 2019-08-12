import pandas as pd
import unicodedata
import re
import string
from pathlib import Path


def preprocess(sentence):
    sentence = re.sub(r'[' + re.escape(string.punctuation) + ']', '', sentence)
    sentence = re.sub(r'\s{2,}', ' ', sentence)
    sentence = unicodedata.normalize('NFKD', sentence)
    sentence = sentence.strip()
    sentence = sentence.lower()
    return sentence


raw_dir = Path.cwd() / 'raw'
list_of_filepaths = sorted(raw_dir.iterdir())


data_dir = Path.cwd() / 'data'
if not data_dir.exists():
    data_dir.mkdir()

ko = []
en = []

for filepath in list_of_filepaths:
    with open(filepath, mode='r', encoding='utf-8') as io:
        corpus = [preprocess(sen) for sen in io.readlines()]

    if 'en' in filepath.suffix:
        en.append(corpus)
    else:
        ko.append(corpus)

for ds in zip(ko, en, ['dev', 'test', 'train']):
    df = pd.DataFrame({'ko': ds[0], 'en': ds[1]})
    filepath = (data_dir / ds[2]).with_suffix('.txt')
    df.to_csv(filepath, sep='\t', index=False)

import pandas as pd
import numpy as np
from pathlib import Path
from model.split import Stemmer

raw_dir = Path.cwd() / 'raw'
list_of_filepath = sorted(raw_dir.iterdir())


data_dir = Path.cwd() / 'data'
if not data_dir.exists():
    data_dir.mkdir()

ko = []
en = []

for filepath in list_of_filepath:
    with open(filepath, mode='r', encoding='utf-8') as io:
        corpus = [sen.strip() for sen in io.readlines()]

    if 'en' in filepath.suffix:
        en.append(corpus)
    else:
        ko.append(corpus)

split_ko = Stemmer(language='ko')
split_en = Stemmer(language='en')

for ds in zip(ko, en, ['dev', 'test', 'train']):
    ko_idx = [idx for idx, sen in enumerate(ds[0]) if len(split_ko.extract_stem(sen)) <= 30]
    en_idx = [idx for idx, sen in enumerate(ds[1]) if len(split_en.extract_stem(sen)) <= 30]
    intersect_idx = set(np.intersect1d(ko_idx, en_idx))
    ko_ds = [sen for idx, sen in enumerate(ds[0]) if idx in intersect_idx]
    en_ds = [sen for idx, sen in enumerate(ds[1]) if idx in intersect_idx]
    df = pd.DataFrame({'ko': ko_ds, 'en': en_ds})
    df = df[(df['ko'].apply(bool) & df['en'].apply(bool))]
    df.to_csv((data_dir / ds[2]).with_suffix('.txt'), index=False, sep='\t')

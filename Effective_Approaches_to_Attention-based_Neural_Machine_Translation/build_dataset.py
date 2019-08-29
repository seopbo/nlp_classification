import pandas as pd
from pathlib import Path


raw_dir = Path.cwd() / 'raw'
list_of_filepath = sorted(raw_dir.iterdir())


data_dir = Path.cwd() / 'data'
if not data_dir.exists():
    data_dir.mkdir()

ko = []
en = []

filepath = list_of_filepath[0]
for filepath in list_of_filepath:
    with open(filepath, mode='r', encoding='utf-8') as io:
        corpus = [sen.strip() for sen in io.readlines()]

    if 'en' in filepath.suffix:
        en.append(corpus)
    else:
        ko.append(corpus)

for ds in zip(ko, en, ['dev', 'test', 'train']):
    df = pd.DataFrame({'ko': ds[0], 'en': ds[1]})
    df = df[(df['ko'].apply(bool) & df['en'].apply(bool))]
    df.to_csv((data_dir / ds[2]).with_suffix('.txt'), index=False, sep='\t')

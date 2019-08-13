import pandas as pd
from model.split import preprocess
from pathlib import Path


raw_dir = Path.cwd() / 'raw'
list_of_filepath = sorted(raw_dir.iterdir())


data_dir = Path.cwd() / 'data'
if not data_dir.exists():
    data_dir.mkdir()

ko = []
en = []

for filepath in list_of_filepath:
    with open(filepath, mode='r', encoding='utf-8') as io:
        corpus = [preprocess(sen) for sen in io.readlines()]

    if 'en' in filepath.suffix:
        en.append(corpus)
    else:
        ko.append(corpus)

tr = pd.DataFrame({'ko': ko[2], 'en': en[2]})
tr = tr[(tr['ko'].apply(bool) & tr['en'].apply(bool))]
tr.to_csv((data_dir / 'train').with_suffix('.csv'), index=False)

dev = pd.DataFrame({'ko': ko[1], 'en': en[1]})
dev = dev[(dev['ko'].apply(bool) & dev['en'].apply(bool))]
dev.to_csv((data_dir / 'dev').with_suffix('.csv'), index=False)

test = pd.DataFrame({'ko': ko[0], 'en': en[0]})
test = test[(test['ko'].apply(bool) & test['en'].apply(bool))]
test.to_csv((data_dir / 'test').with_suffix('.csv'), index=False)

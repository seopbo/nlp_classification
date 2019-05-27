import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# loading dataset
proj_dir = Path.cwd()
filepath = proj_dir / 'data' / 'ratings_train.txt'
dataset = pd.read_csv(filepath, sep='\t').loc[:, ['document', 'label']]
dataset = dataset.loc[dataset['document'].isna().apply(lambda elm: not elm), :]
tr, val = train_test_split(dataset, test_size=0.2, random_state=777)

tr.to_csv(proj_dir / 'data' / 'train.txt', sep='\t', index=False, header=False)
val.to_csv(proj_dir / 'data' / 'val.txt', sep='\t', index=False, header=False)

tst_filepath = proj_dir / 'data' / 'ratings_test.txt'
tst = pd.read_csv(tst_filepath, sep='\t').loc[:, ['document', 'label']]
tst = tst.loc[tst['document'].isna().apply(lambda elm: not elm), :]
tst.to_csv(proj_dir / 'data' / 'test.txt', sep='\t', index=False, header=False)
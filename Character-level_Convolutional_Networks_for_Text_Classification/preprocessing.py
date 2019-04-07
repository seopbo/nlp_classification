import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from model.utils import JamoTokenizer

# loading dataset
proj_dir = Path('.')
tr_filepath = proj_dir / 'data/ratings_train.txt'
data = pd.read_csv(tr_filepath, sep='\t').loc[:, ['document', 'label']]
data = data.loc[data['document'].isna().apply(lambda elm: not elm), :]

tokenizer = JamoTokenizer()

comment_length = data['document'].apply(lambda comment: len(tokenizer.tokenize(comment)))

comment_length.hist()
plt.show()

tr_data, val_data = train_test_split(data, test_size=.2)

tst_filepath = proj_dir / 'data/ratings_test.txt'
tst_data = pd.read_csv(tst_filepath, sep='\t').loc[:, ['document', 'label']]
tst_data = tst_data.loc[tst_data['document'].isna().apply(lambda elm: not elm), :]

# saving tr_data, val_data, tst_data
tr_data.to_csv('./data/train.txt', index=False, sep="\t")
val_data.to_csv('./data/val.txt', index=False, sep="\t")
tst_data.to_csv('./data/test.txt', index=False, sep="\t")
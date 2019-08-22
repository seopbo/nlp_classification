import pickle
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from pretrained.tokenization import BertTokenizer

# dataset
cwd = Path.cwd()
dataset = pd.read_csv(cwd / 'data' / 'kor_pair_train.csv').filter(items=['question1', 'question2', 'is_duplicate'])
tst = pd.read_csv(cwd / 'data' / 'kor_pair_test.csv').filter(items=['question1', 'question2', 'is_duplicate'])
total = pd.concat([dataset, tst], axis=0, ignore_index=True, sort=False)
total['is_duplicate']
total.iloc[:30]
tr, val = train_test_split(total, test_size=.2, random_state=777)

# eda (length)
ptr_tokenizer = BertTokenizer.from_pretrained('pretrained/vocab.korean.rawtext.list', do_lower_case=False)
with open('pretrained/vocab.pkl', mode='rb') as io:
    vocab = pickle.load(io)

question1_length = tr['question1'].apply(lambda sen: len(ptr_tokenizer.tokenize(sen)))
question2_length = tr['question2'].apply(lambda sen: len(ptr_tokenizer.tokenize(sen)))
(question1_length + question2_length).describe()

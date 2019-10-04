import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# dataset
cwd = Path.cwd()
dataset = pd.read_csv(cwd / "data" / "kor_pair_train.csv").filter(
    items=["question1", "question2", "is_duplicate"]
)
tst = pd.read_csv(cwd / "data" / "kor_pair_test.csv").filter(
    items=["question1", "question2", "is_duplicate"]
)

total = pd.concat([dataset, tst], axis=0, ignore_index=True, sort=False)
train, validation = train_test_split(total, test_size=0.1, random_state=777)

train.to_csv(cwd / "data" / "train.txt", sep="\t", index=False)
validation.to_csv(cwd / "data" / "validation.txt", sep="\t", index=False)

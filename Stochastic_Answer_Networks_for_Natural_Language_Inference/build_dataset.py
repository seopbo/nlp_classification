import pandas as pd
from pathlib import Path
from utils import Config
from sklearn.model_selection import train_test_split

# dataset
qpair_dir = Path("qpair")
train = pd.read_csv(qpair_dir / "kor_pair_train.csv").filter(
    items=["question1", "question2", "is_duplicate"]
)

test = pd.read_csv(qpair_dir / "kor_pair_test.csv").filter(
    items=["question1", "question2", "is_duplicate"]
)

dataset = pd.concat([train, test], ignore_index=True, sort=False)

train, test = train_test_split(dataset, test_size=.1, random_state=777)
train, validation = train_test_split(train, test_size=.1, random_state=777)

train.to_csv(qpair_dir / "train.txt", sep="\t", index=False)
validation.to_csv(qpair_dir / "validation.txt", sep="\t", index=False)
test.to_csv(qpair_dir / "test.txt", sep="\t", index=False)

config = Config(
    {
        "train": str(qpair_dir / "train.txt"),
        "validation": str(qpair_dir / "validation.txt"),
        "test": str(qpair_dir / "test.txt"),
    }
)

config.save("conf/dataset/qpair.json")

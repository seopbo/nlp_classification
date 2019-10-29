import pandas as pd
from pathlib import Path
from utils import Config
from sklearn.model_selection import train_test_split

# dataset
data_dir = Path("data")
train = pd.read_csv(data_dir / "kor_pair_train.csv").filter(
    items=["question1", "question2", "is_duplicate"]
)

test = pd.read_csv(data_dir / "kor_pair_test.csv").filter(
    items=["question1", "question2", "is_duplicate"]
)

dataset = pd.concat([train, test], ignore_index=True, sort=False)

train, test = train_test_split(dataset, test_size=.1, random_state=777)
train, validation = train_test_split(train, test_size=.1, random_state=777)

train.to_csv(data_dir / "train.txt", sep="\t", index=False)
validation.to_csv(data_dir / "validation.txt", sep="\t", index=False)
test.to_csv(data_dir / "test.txt", sep="\t", index=False)

config = Config(
    {
        "train": str(data_dir / "train.txt"),
        "validation": str(data_dir / "validation.txt"),
        "test": str(data_dir / "test.txt"),
    }
)
config.save(data_dir / "config.json")

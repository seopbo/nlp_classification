import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from utils import Config

# loading dataset
data_dir = Path("data")
filepath = data_dir / "ratings_train.txt"
dataset = pd.read_csv(filepath, sep="\t").loc[:, ["document", "label"]]
dataset = dataset.loc[dataset["document"].isna().apply(lambda elm: not elm), :]
tr, val = train_test_split(dataset, test_size=0.2, random_state=777)

tr.to_csv(data_dir / "train.txt", sep="\t", index=False)
val.to_csv(data_dir / "validation.txt", sep="\t", index=False)

tst_filepath = data_dir / "ratings_test.txt"
tst = pd.read_csv(tst_filepath, sep="\t").loc[:, ["document", "label"]]
tst = tst.loc[tst["document"].isna().apply(lambda elm: not elm), :]
tst.to_csv(data_dir / "test.txt", sep="\t", index=False)

config = Config(
    {
        "train": str(data_dir / "train.txt"),
        "validation": str(data_dir / "validation.txt"),
        "test": str(data_dir / "test.txt"),
    }
)
config.save(data_dir / "config.json")
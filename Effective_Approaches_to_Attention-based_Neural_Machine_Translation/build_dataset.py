import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from utils import Config

sample_dir = Path("sample")
list_of_filepath = sorted(sample_dir.iterdir())

data = pd.read_excel(list_of_filepath[0])
data = data.loc[:, data.columns[[1, 3]]]
data.columns = ["ko", "en"]

tr, val = train_test_split(data, test_size=0.1, random_state=777)
val, tst = train_test_split(val, test_size=0.1, random_state=777)

tr_filepath = (sample_dir / "train").with_suffix(".txt")
val_filepath = (sample_dir / "validation").with_suffix(".txt")
tst_filepath = (sample_dir / "test").with_suffix(".txt")

tr.to_csv(tr_filepath, sep="\t", index=False)
val.to_csv(val_filepath, sep="\t", index=False)
tst.to_csv(tst_filepath, sep="\t", index=False)

data_config = Config(
    {
        "train": str(tr_filepath),
        "validation": str(val_filepath),
        "test": str(tst_filepath),
    }
)

data_config.save("conf/dataset/sample.json")

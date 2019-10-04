import argparse
import pickle
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from model.net import BilstmCRF
from model.data import Corpus, batchify
from model.utils import Tokenizer
from model.split import split_to_self
from model.metric import get_f1_score
from utils import Config, CheckpointManager, SummaryManager


parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir", default="data", help="Directory containing config.json of data"
)
parser.add_argument(
    "--model_dir",
    default="experiments/base_model",
    help="Directory containing config.json of model",
)
parser.add_argument(
    "--restore_file",
    default="best",
    help="name of the file in --model_dir \
                     containing weights to load",
)
parser.add_argument(
    "--data_name", default="test", help="name of the data in --data_dir to be evaluate"
)


if __name__ == "__main__":
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    data_config = Config(json_path=data_dir / "config.json")
    model_config = Config(json_path=model_dir / "config.json")

    # tokenizer
    with open(data_config.token_vocab, mode="rb") as io:
        token_vocab = pickle.load(io)
    with open(data_config.label_vocab, mode="rb") as io:
        label_vocab = pickle.load(io)
    token_tokenizer = Tokenizer(token_vocab, split_to_self)
    label_tokenizer = Tokenizer(label_vocab, split_to_self)

    # model (restore)
    checkpoint_manager = CheckpointManager(model_dir)
    checkpoint = checkpoint_manager.load_checkpoint(args.restore_file + ".tar")
    model = BilstmCRF(label_vocab, token_vocab, model_config.lstm_hidden_dim)
    model.load_state_dict(checkpoint["model_state_dict"])

    # evaluation
    summary_manager = SummaryManager(model_dir)
    filepath = getattr(data_config, args.data_name)
    ds = Corpus(
        filepath,
        token_tokenizer.split_and_transform,
        label_tokenizer.split_and_transform,
    )
    dl = DataLoader(
        ds, batch_size=model_config.batch_size, num_workers=4, collate_fn=batchify
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    f1_score = get_f1_score(model, dl, device)
    summary_manager.load("summary.json")
    summary_manager._summary[args.data_name].update({"f1": f1_score})
    summary_manager.save("summary.json")

    print("f1_score: {:.2%}".format(f1_score))

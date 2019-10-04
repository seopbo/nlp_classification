import argparse
import pickle
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from torch.utils.data import DataLoader
from model.net import BilstmCRF
from model.data import Corpus, batchify
from model.utils import Tokenizer
from model.split import split_to_self
from model.metric import evaluate
from utils import Config, CheckpointManager, SummaryManager
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir", default="data", help="Directory containing config.json of data"
)
parser.add_argument(
    "--model_dir",
    default="experiments/base_model",
    help="Directory containing config.json of model",
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

    # model
    model = BilstmCRF(label_vocab, token_vocab, model_config.lstm_hidden_dim)

    # training
    tr_ds = Corpus(
        data_config.train,
        token_tokenizer.split_and_transform,
        label_tokenizer.split_and_transform,
    )
    tr_dl = DataLoader(
        tr_ds,
        batch_size=model_config.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        collate_fn=batchify,
    )
    val_ds = Corpus(
        data_config.validation,
        token_tokenizer.split_and_transform,
        label_tokenizer.split_and_transform,
    )
    val_dl = DataLoader(
        val_ds, batch_size=model_config.batch_size, num_workers=4, collate_fn=batchify
    )

    opt = optim.Adam(params=model.parameters(), lr=model_config.learning_rate)
    scheduler = ReduceLROnPlateau(opt, patience=5)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    writer = SummaryWriter("{}/runs".format(model_dir))
    checkpoint_manager = CheckpointManager(model_dir)
    summary_manager = SummaryManager(model_dir)
    best_val_loss = 1e10

    for epoch in tqdm(range(model_config.epochs), desc="epochs"):

        tr_loss = 0

        model.train()
        for step, mb in tqdm(enumerate(tr_dl), desc="steps", total=len(tr_dl)):
            x_mb, y_mb = map(lambda elm: elm.to(device), mb)

            opt.zero_grad()
            mb_loss = model.loss(x_mb, y_mb)
            mb_loss.backward()
            opt.step()

            tr_loss += mb_loss.item()

            if (epoch * len(tr_dl) + step) % model_config.summary_step == 0:
                val_loss = evaluate(model, val_dl, device)
                writer.add_scalars(
                    "loss",
                    {"train": tr_loss / (step + 1), "val": val_loss},
                    epoch * len(tr_dl) + step,
                )
                model.train()
        else:
            tr_loss /= step + 1

            tr_summary = {"loss": tr_loss}
            val_loss = evaluate(model, val_dl, device)
            scheduler.step(val_loss)
            tqdm.write(
                "epoch : {}, tr_loss: {:.3f}, val_loss: {:.3f}".format(
                    epoch + 1, tr_summary["loss"], val_loss
                )
            )
            is_best = val_loss < best_val_loss
            val_summary = {"loss": val_loss}

            if is_best:
                state = {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "opt_state_dict": opt.state_dict(),
                }
                summary = {"train": tr_summary, "validation": val_summary}

                summary_manager.update(summary)
                summary_manager.save("summary.json")
                checkpoint_manager.save_checkpoint(state, "best.tar")

                best_val_loss = val_loss

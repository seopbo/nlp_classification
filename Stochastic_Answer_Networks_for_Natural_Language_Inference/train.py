import argparse
import pickle
import torch
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader
from model.net import SAN
from model.data import Corpus, batchify
from model.utils import PreProcessor
from model.split import split_morphs, split_jamos
from model.metric import evaluate, acc, log_loss
from utils import Config, CheckpointManager, SummaryManager
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def get_preprocessor(dataset_config, coarse_split_fn, fine_split_fn):
    with open(dataset_config.fine_vocab, mode="rb") as io:
        fine_vocab = pickle.load(io)
    with open(dataset_config.coarse_vocab, mode="rb") as io:
        coarse_vocab = pickle.load(io)

    preprocessor = PreProcessor(coarse_vocab=coarse_vocab, fine_vocab=fine_vocab,
                                coarse_split_fn=coarse_split_fn,
                                fine_split_fn=fine_split_fn)
    return preprocessor


def get_data_loaders(dataset_config, preprocessor, batch_size, collate_fn):
    tr_ds = Corpus(dataset_config.train, preprocessor.preprocess)
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True,
                       collate_fn=collate_fn)
    val_ds = Corpus(dataset_config.validation, preprocessor.preprocess)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=4, collate_fn=collate_fn)
    return tr_dl, val_dl


def main(args):
    dataset_config = Config(args.dataset_config)
    model_config = Config(args.model_config)

    exp_dir = Path("experiments") / model_config.type
    exp_dir = exp_dir.joinpath(
        f"epochs_{args.epochs}_batch_size_{args.batch_size}_learning_rate_{args.learning_rate}"
    )

    if not exp_dir.exists():
        exp_dir.mkdir(parents=True)

    if args.fix_seed:
        torch.manual_seed(777)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    preprocessor = get_preprocessor(dataset_config, coarse_split_fn=split_morphs, fine_split_fn=split_jamos)
    tr_dl, val_dl = get_data_loaders(dataset_config, preprocessor, args.batch_size, collate_fn=batchify)

    # model
    model = SAN(model_config.num_classes, preprocessor.coarse_vocab, preprocessor.fine_vocab,
                model_config.fine_embedding_dim, model_config.hidden_dim, model_config.multi_step,
                model_config.prediction_drop_ratio)

    opt = optim.Adam(model.parameters(), lr=args.learning_rate)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    writer = SummaryWriter(f"{exp_dir}/runs")
    checkpoint_manager = CheckpointManager(exp_dir)
    summary_manager = SummaryManager(exp_dir)
    best_val_loss = 1e10

    for epoch in tqdm(range(args.epochs), desc="epochs"):

        tr_loss = 0
        tr_acc = 0

        model.train()
        for step, mb in tqdm(enumerate(tr_dl), desc="steps", total=len(tr_dl)):
            qa_mb, qb_mb, y_mb = map(lambda elm: (el.to(device) for el in elm) if isinstance(elm, tuple) else
            elm.to(device), mb)
            opt.zero_grad()
            y_hat_mb = model((qa_mb, qb_mb))
            mb_loss = log_loss(y_hat_mb, y_mb)
            mb_loss.backward()
            opt.step()

            with torch.no_grad():
                mb_acc = acc(y_hat_mb, y_mb)

            tr_loss += mb_loss.item()
            tr_acc += mb_acc.item()

            if (epoch * len(tr_dl) + step) % args.summary_step == 0:
                val_loss = evaluate(model, val_dl, {"loss": log_loss}, device)["loss"]
                writer.add_scalars("loss", {"train": tr_loss / (step + 1), "val": val_loss}, epoch * len(tr_dl) + step)
                tqdm.write(f"global_step: {epoch * len(tr_dl) + step:3}\n"
                           f"tr_loss: {tr_loss / (step + 1):.3f}, "
                           f"val_loss: {val_loss:.3f}")
                model.train()
        else:
            tr_loss /= step + 1
            tr_acc /= step + 1

            tr_summary = {"loss": tr_loss, "acc": tr_acc}
            val_summary = evaluate(model, val_dl, {"loss": log_loss, "acc": acc}, device)
            tqdm.write(f"epoch : {epoch + 1}\n"
                       f"tr_loss: {tr_summary['loss']:.3f}, "
                       f"val_loss: {val_summary['loss']:.3f}\n"
                       f"tr_acc: {tr_summary['acc']:.2%}, "
                       f"val_acc: {val_summary['acc']:.2%}")

            val_loss = val_summary["loss"]
            is_best = val_loss < best_val_loss

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_config", default="conf/dataset/qpair.json")
    parser.add_argument("--model_config", default="conf/model/stochastic.json")
    parser.add_argument("--epochs", default=5, help="number of epochs of training")
    parser.add_argument("--batch_size", default=128, help="batch size of training")
    parser.add_argument("--learning_rate", default=1e-3, help="learning rate of training")
    parser.add_argument("--summary_step", default=500, help="logging performance at each step")
    parser.add_argument("--fix_seed", action="store_true", default=False)
    args = parser.parse_args()
    main(args)

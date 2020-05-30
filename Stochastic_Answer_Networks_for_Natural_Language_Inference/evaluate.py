import argparse
import pickle
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from model.net import SAN
from model.data import Corpus, batchify
from model.utils import PreProcessor
from model.split import split_morphs, split_jamos
from model.metric import evaluate, acc, log_loss
from utils import Config, CheckpointManager, SummaryManager


def get_preprocessor(dataset_config, coarse_split_fn, fine_split_fn):
    with open(dataset_config.fine_vocab, mode="rb") as io:
        fine_vocab = pickle.load(io)
    with open(dataset_config.coarse_vocab, mode="rb") as io:
        coarse_vocab = pickle.load(io)

    preprocessor = PreProcessor(coarse_vocab=coarse_vocab, fine_vocab=fine_vocab,
                                coarse_split_fn=coarse_split_fn,
                                fine_split_fn=fine_split_fn)
    return preprocessor


def main(args):
    dataset_config = Config(args.dataset_config)
    model_config = Config(args.model_config)

    exp_dir = Path("experiments") / model_config.type
    exp_dir = exp_dir.joinpath(
        f"epochs_{args.epochs}_batch_size_{args.batch_size}_learning_rate_{args.learning_rate}"
    )

    preprocessor = get_preprocessor(dataset_config, coarse_split_fn=split_morphs, fine_split_fn=split_jamos)

    # model (restore)
    checkpoint_manager = CheckpointManager(exp_dir)
    checkpoint = checkpoint_manager.load_checkpoint("best.tar")
    model = SAN(model_config.num_classes, preprocessor.coarse_vocab, preprocessor.fine_vocab,
                model_config.fine_embedding_dim, model_config.hidden_dim, model_config.multi_step,
                model_config.prediction_drop_ratio)
    model.load_state_dict(checkpoint["model_state_dict"])

    # evaluation
    filepath = getattr(dataset_config, args.data)
    ds = Corpus(filepath, preprocessor.preprocess)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=4, collate_fn=batchify)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    summary_manager = SummaryManager(exp_dir)
    summary = evaluate(model, dl, {"loss": log_loss, "acc": acc}, device)

    summary_manager.load("summary.json")
    summary_manager.update({f"{args.data}": summary})
    summary_manager.save("summary.json")

    print(f"loss: {summary['loss']:.3f}, acc: {summary['acc']:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_config", default="conf/dataset/qpair.json")
    parser.add_argument("--data", default="test", help="name of the data in qpair_dir to be evaluate")
    parser.add_argument("--model_config", default="conf/model/stochastic.json")
    parser.add_argument("--epochs", default=5, help="number of epochs of training")
    parser.add_argument("--batch_size", default=128, help="batch size of training")
    parser.add_argument("--learning_rate", default=1e-3, help="learning rate of training")
    args = parser.parse_args()
    main(args)

import argparse
import torch
import torch.nn as nn
import pickle
from pathlib import Path
from torch.utils.data import DataLoader
from model.split import split_to_jamo
from model.data import Corpus, batchify
from model.net import ConvRec
from model.utils import Tokenizer
from model.metric import evaluate, acc
from utils import Config, CheckpointManager, SummaryManager


def get_tokenizer(dataset_config):
    with open(dataset_config.vocab, mode='rb') as io:
        vocab = pickle.load(io)
    tokenizer = Tokenizer(vocab=vocab, split_fn=split_to_jamo)
    return tokenizer


def main(args):
    dataset_config = Config(args.dataset_config)
    model_config = Config(args.model_config)

    exp_dir = Path("experiments") / model_config.type
    exp_dir = exp_dir.joinpath(
        f"epochs_{args.epochs}_batch_size_{args.batch_size}_learning_rate_{args.learning_rate}"
    )

    tokenizer = get_tokenizer(dataset_config)

    checkpoint_manager = CheckpointManager(exp_dir)
    checkpoint = checkpoint_manager.load_checkpoint("best.tar")
    model = ConvRec(num_classes=model_config.num_classes, embedding_dim=model_config.embedding_dim,
                    hidden_dim=model_config.hidden_dim, vocab=tokenizer.vocab)
    model.load_state_dict(checkpoint["model_state_dict"])

    summary_manager = SummaryManager(exp_dir)
    filepath = getattr(dataset_config, args.data)
    ds = Corpus(filepath, tokenizer.split_and_transform, min_length=model_config.min_length,
                pad_val=tokenizer.vocab.to_indices(' '))
    dl = DataLoader(ds, batch_size=args.batch_size, collate_fn=batchify)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    summary = evaluate(model, dl, {"loss": nn.CrossEntropyLoss(), "acc": acc}, device)

    summary_manager.load("summary.json")
    summary_manager.update({f"{args.data}": summary})
    summary_manager.save("summary.json")
    print(f"loss: {summary['loss']:.3f}, acc: {summary['acc']:.2%}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default="test", help="name of the data in nsmc_dir to be evaluate"
    )
    parser.add_argument(
        "--dataset_config",
        default="conf/dataset/nsmc.json",
        help="directory containing nsmc.json",
    )
    parser.add_argument(
        "--model_config",
        default="conf/model/convrec.json",
        help="directory containing charcnn.json",
    )
    parser.add_argument("--epochs", default=5, help="number of epochs of training")
    parser.add_argument("--batch_size", default=256, help="batch size of training")
    parser.add_argument(
        "--learning_rate", default=1e-3, help="learning rate of training"
    )

    args = parser.parse_args()
    main(args)

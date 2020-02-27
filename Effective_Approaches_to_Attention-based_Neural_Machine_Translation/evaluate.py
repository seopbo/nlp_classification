import argparse
import pickle
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from model.split import Stemmer
from model.net import BidiEncoder, AttnDecoder
from model.data import NMTCorpus, batchify
from model.utils import SourceProcessor, TargetProcessor
from model.metric import evaluate
from utils import Config, CheckpointManager, SummaryManager


def get_processor(dataset_config):
    with open(dataset_config.source_vocab, mode="rb") as io:
        src_vocab = pickle.load(io)
    src_stemmer = Stemmer(language="ko")
    src_processor = SourceProcessor(src_vocab, src_stemmer.extract_stem)

    with open(dataset_config.target_vocab, mode="rb") as io:
        tgt_vocab = pickle.load(io)
    tgt_stemmer = Stemmer(language="en")
    tgt_processor = TargetProcessor(tgt_vocab, tgt_stemmer.extract_stem)
    return src_processor, tgt_processor


def main(args):
    dataset_config = Config(args.dataset_config)
    model_config = Config(args.model_config)

    exp_dir = Path("experiments") / model_config.type
    exp_dir = exp_dir.joinpath(
        f"epochs_{args.epochs}_batch_size_{args.batch_size}_learning_rate_{args.learning_rate}"
        f"_teacher_forcing_ratio_{args.teacher_forcing_ratio}"
    )

    src_processor, tgt_processor = get_processor(dataset_config)

    # model (restore)
    encoder = BidiEncoder(
        src_processor.vocab, model_config.encoder_hidden_dim, model_config.drop_ratio
    )
    decoder = AttnDecoder(
        tgt_processor.vocab,
        model_config.method,
        model_config.encoder_hidden_dim * 2,
        model_config.decoder_hidden_dim,
        model_config.drop_ratio,
    )

    checkpoint_manager = CheckpointManager(exp_dir)
    checkpoint = checkpoint_manager.load_checkpoint("best.tar")
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])

    encoder.eval()
    decoder.eval()

    # evaluation
    summary_manager = SummaryManager(exp_dir)
    filepath = getattr(dataset_config, args.data)
    ds = NMTCorpus(filepath, src_processor.process, tgt_processor.process)
    dl = DataLoader(
        ds,
        args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=batchify,
        drop_last=False,
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    encoder.to(device)
    decoder.to(device)

    loss = evaluate(encoder, decoder, tgt_processor.vocab, dl, device)
    summary = {"perplexity": np.exp(loss)}
    summary_manager.load("summary.json")
    summary_manager.update({"{}".format(args.data): summary})
    summary_manager.save("summary.json")
    print("perplexity: {:.3f}".format(np.exp(loss)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="test",
        help="name of the data in sample directory to be evaluate",
    )
    parser.add_argument(
        "--dataset_config",
        default="conf/dataset/sample.json",
        help="directory containing sample.json",
    )
    parser.add_argument(
        "--model_config",
        default="conf/model/luongattn.json",
        help="directory containing luongattn.json",
    )
    parser.add_argument("--epochs", default=50, help="number of epochs of training")
    parser.add_argument("--batch_size", default=256, help="batch size of training")
    parser.add_argument(
        "--learning_rate", default=1e-3, help="learning rate of training"
    )
    parser.add_argument(
        "--teacher_forcing_ratio", default=0.1, help="teacher forcing ratio of training"
    )
    args = parser.parse_args()

    main(args)

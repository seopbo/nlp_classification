import argparse
import pickle
import torch
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from model.split import Stemmer
from model.net import BidiEncoder, AttnDecoder
from model.data import NMTCorpus, batchify
from model.utils import SourceProcessor, TargetProcessor
from model.metric import mask_nll_loss, sequence_mask, evaluate
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


def get_data_loaders(dataset_config, src_processor, tgt_processor, batch_size, collate_fn=batchify):
    tr_ds = NMTCorpus(
        dataset_config.train, src_processor.process, tgt_processor.process
    )
    tr_dl = DataLoader(
        tr_ds,
        batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_ds = NMTCorpus(
        dataset_config.validation, src_processor.process, tgt_processor.process
    )
    val_dl = DataLoader(
        val_ds,
        batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        drop_last=False,
    )
    return tr_dl, val_dl


def main(args):
    dataset_config = Config(args.dataset_config)
    model_config = Config(args.model_config)

    exp_dir = Path("experiments") / model_config.type
    exp_dir = exp_dir.joinpath(
        f"epochs_{args.epochs}_batch_size_{args.batch_size}_learning_rate_{args.learning_rate}"
        f"_teacher_forcing_ratio_{args.teacher_forcing_ratio}"
    )

    if not exp_dir.exists():
        exp_dir.mkdir(parents=True)

    if args.fix_seed:
        torch.manual_seed(777)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # processor
    src_processor, tgt_processor = get_processor(dataset_config)

    # data_loaders
    tr_dl, val_dl = get_data_loaders(
        dataset_config, src_processor, tgt_processor, args.batch_size
    )

    # model
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

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    encoder.to(device)
    decoder.to(device)

    writer = SummaryWriter("{}/runs".format(exp_dir))
    checkpoint_manager = CheckpointManager(exp_dir)
    summary_manager = SummaryManager(exp_dir)
    best_val_loss = 1e10

    opt = optim.RMSprop(
        [{"params": encoder.parameters()}, {"params": decoder.parameters()}],
        lr=args.learning_rate,
    )
    scheduler = ReduceLROnPlateau(opt, patience=5)

    for epoch in tqdm(range(args.epochs), desc="epochs"):
        tr_loss = 0

        encoder.train()
        decoder.train()

        for step, mb in tqdm(enumerate(tr_dl), desc="steps", total=len(tr_dl)):
            mb_loss = 0
            src_mb, tgt_mb = map(lambda elm: elm.to(device), mb)
            opt.zero_grad()

            # encoder
            enc_outputs_mb, src_length_mb, enc_hc_mb = encoder(src_mb)

            # decoder
            dec_input_mb = torch.ones((tgt_mb.size()[0], 1), device=device).long()
            dec_input_mb *= tgt_processor.vocab.to_indices(
                tgt_processor.vocab.bos_token
            )
            dec_hc_mb = None
            tgt_length_mb = tgt_mb.ne(
                tgt_processor.vocab.to_indices(tgt_processor.vocab.padding_token)
            ).sum(dim=1)
            tgt_mask_mb = sequence_mask(tgt_length_mb, tgt_length_mb.max())

            use_teacher_forcing = (
                True if random.random() > args.teacher_forcing_ratio else False
            )

            if use_teacher_forcing:
                for t in range(tgt_length_mb.max()):
                    dec_output_mb, dec_hc_mb = decoder(
                        dec_input_mb, dec_hc_mb, enc_outputs_mb, src_length_mb
                    )
                    sequence_loss = mask_nll_loss(
                        dec_output_mb, tgt_mb[:, [t]], tgt_mask_mb[:, [t]]
                    )
                    mb_loss += sequence_loss
                    dec_input_mb = tgt_mb[:, [t]]  # next input is current target
                else:
                    mb_loss /= tgt_length_mb.max()
            else:
                for t in range(tgt_length_mb.max()):
                    dec_output_mb, dec_hc_mb = decoder(
                        dec_input_mb, dec_hc_mb, enc_outputs_mb, src_length_mb
                    )
                    sequence_loss = mask_nll_loss(
                        dec_output_mb, tgt_mb[:, [t]], tgt_mask_mb[:, [t]]
                    )
                    mb_loss += sequence_loss
                    dec_input_mb = dec_output_mb.topk(1).indices
                else:
                    mb_loss /= tgt_length_mb.max()

            # update params
            mb_loss.backward()
            nn.utils.clip_grad_norm_(encoder.parameters(), args.clip_norm)
            nn.utils.clip_grad_norm_(decoder.parameters(), args.clip_norm)
            opt.step()

            tr_loss += mb_loss.item()

            if (epoch * len(tr_dl) + step) % args.summary_step == 0:
                val_loss = evaluate(
                    encoder, decoder, tgt_processor.vocab, val_dl, device
                )
                writer.add_scalars(
                    "perplexity",
                    {"train": np.exp(tr_loss / (step + 1)), "validation": np.exp(val_loss)},
                    epoch * len(tr_dl) + step,
                )
                encoder.train()
                decoder.train()

        else:
            tr_loss /= step + 1

            tr_summary = {"perplexity": np.exp(tr_loss)}
            val_loss = evaluate(encoder, decoder, tgt_processor.vocab, val_dl, device)
            val_summary = {"perplexity": np.exp(val_loss)}
            scheduler.step(np.exp(val_loss))

            tqdm.write(
                "epoch : {}, tr_ppl: {:.3f}, val_ppl: "
                "{:.3f}".format(epoch + 1, tr_summary["perplexity"], val_summary["perplexity"])
            )

            is_best = val_loss < best_val_loss

            if is_best:
                state = {
                    "epoch": epoch + 1,
                    "encoder_state_dict": encoder.state_dict(),
                    "decoder_state_dict": decoder.state_dict(),
                    "opt_state_dict": opt.state_dict(),
                }
                summary = {
                    "epoch": epoch + 1,
                    "train": tr_summary,
                    "validation": val_summary,
                }

                summary_manager.update(summary)
                summary_manager.save("summary.json")
                checkpoint_manager.save_checkpoint(state, "best.tar")

                best_val_loss = val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--clip_norm", default=5, help="clip_norm of training")
    parser.add_argument(
        "--summary_step", default=500, help="logging performance at each step"
    )
    parser.add_argument("--fix_seed", action="store_true", default=False)

    args = parser.parse_args()
    main(args)

import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import json
from pathlib import Path
from torch.utils.data import DataLoader
from transformers.modeling_bert import BertConfig
from model.tokenization import BertTokenizer as ETRITokenizer
from gluonnlp.data import SentencepieceTokenizer
from model.net import PairwiseClassifier
from model.data import Corpus
from model.utils import PreProcessor, PadSequence
from model.metric import evaluate, acc
from utils import Config, CheckpointManager, SummaryManager
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def get_preprocessor(ptr_config_info, model_config):
    with open(ptr_config_info.vocab, mode='rb') as io:
        vocab = pickle.load(io)

    if model_config.type == 'etri':
        ptr_tokenizer = ETRITokenizer.from_pretrained(ptr_config_info.tokenizer, do_lower_case=False)
        pad_sequence = PadSequence(length=model_config.length, pad_val=vocab.to_indices(vocab.padding_token))
        preprocessor = PreProcessor(vocab=vocab, split_fn=ptr_tokenizer.tokenize, pad_fn=pad_sequence)
    elif model_config.type == 'skt':
        ptr_tokenizer = SentencepieceTokenizer(ptr_config_info.tokenizer)
        pad_sequence = PadSequence(length=model_config.length, pad_val=vocab.to_indices(vocab.padding_token))
        preprocessor = PreProcessor(vocab=vocab, split_fn=ptr_tokenizer, pad_fn=pad_sequence)
    return preprocessor


def get_data_loaders(dataset_config, preprocessor, batch_size):
    tr_ds = Corpus(dataset_config.train, preprocessor.preprocess)
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_ds = Corpus(dataset_config.validation, preprocessor.preprocess)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=4)
    return tr_dl, val_dl


def main(args):
    dataset_config = Config(args.dataset_config)
    model_config = Config(args.model_config)
    ptr_config_info = Config(f"conf/pretrained/{model_config.type}.json")

    exp_dir = Path("experiments") / model_config.type
    exp_dir = exp_dir.joinpath(
        f"epochs_{args.epochs}_batch_size_{args.batch_size}_learning_rate_{args.learning_rate}"
        f"_weight_decay_{args.weight_decay}"
    )

    if not exp_dir.exists():
        exp_dir.mkdir(parents=True)

    if args.fix_seed:
        torch.manual_seed(777)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    preprocessor = get_preprocessor(ptr_config_info, model_config)

    with open(ptr_config_info.config, mode="r") as io:
        ptr_config = json.load(io)

    # model
    config = BertConfig()
    config.update(ptr_config)
    model = PairwiseClassifier(config, num_classes=model_config.num_classes, vocab=preprocessor.vocab)
    bert_pretrained = torch.load(ptr_config_info.bert)
    model.load_state_dict(bert_pretrained, strict=False)

    tr_dl, val_dl = get_data_loaders(dataset_config, preprocessor, args.batch_size)

    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(
        [
            {"params": model.bert.parameters(), "lr": args.learning_rate / 100},
            {"params": model.classifier.parameters(), "lr": args.learning_rate},

        ], weight_decay=args.weight_decay)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    writer = SummaryWriter(f'{exp_dir}/runs')
    checkpoint_manager = CheckpointManager(exp_dir)
    summary_manager = SummaryManager(exp_dir)
    best_val_loss = 1e+10

    for epoch in tqdm(range(args.epochs), desc='epochs'):

        tr_loss = 0
        tr_acc = 0

        model.train()
        for step, mb in tqdm(enumerate(tr_dl), desc='steps', total=len(tr_dl)):
            x_mb, x_types_mb, y_mb = map(lambda elm: elm.to(device), mb)
            opt.zero_grad()
            y_hat_mb = model(x_mb, x_types_mb)
            mb_loss = loss_fn(y_hat_mb, y_mb)
            mb_loss.backward()
            opt.step()

            with torch.no_grad():
                mb_acc = acc(y_hat_mb, y_mb)

            tr_loss += mb_loss.item()
            tr_acc += mb_acc.item()

            if (epoch * len(tr_dl) + step) % args.summary_step == 0:
                val_loss = evaluate(model, val_dl, {'loss': loss_fn}, device)['loss']
                writer.add_scalars('loss', {'train': tr_loss / (step + 1),
                                            'val': val_loss}, epoch * len(tr_dl) + step)
                model.train()
        else:
            tr_loss /= (step + 1)
            tr_acc /= (step + 1)

            tr_summary = {'loss': tr_loss, 'acc': tr_acc}
            val_summary = evaluate(model, val_dl, {'loss': loss_fn, 'acc': acc}, device)
            tqdm.write(f"epoch: {epoch+1}\n"
                       f"tr_loss: {tr_summary['loss']:.3f}, val_loss: {val_summary['loss']:.3f}\n"
                       f"tr_acc: {tr_summary['acc']:.2%}, val_acc: {val_summary['acc']:.2%}")

            val_loss = val_summary['loss']
            is_best = val_loss < best_val_loss

            if is_best:
                state = {'epoch': epoch + 1,
                         'model_state_dict': model.state_dict(),
                         'opt_state_dict': opt.state_dict()}
                summary = {'train': tr_summary, 'validation': val_summary}

                summary_manager.update(summary)
                summary_manager.save('summary.json')
                checkpoint_manager.save_checkpoint(state, 'best.tar')

                best_val_loss = val_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_config", default="conf/dataset/qpair.json")
    parser.add_argument("--model_config", default="conf/model/pairwise_skt.json")
    parser.add_argument("--epochs", default=3, help="number of epochs of training")
    parser.add_argument("--batch_size", default=64, help="batch size of training")
    parser.add_argument(
        "--learning_rate", default=1e-3, help="learning rate of training"
    )
    parser.add_argument(
        "--weight_decay", default=5e-4, help="weight decay of training"
    )
    parser.add_argument(
        "--summary_step", default=1000, help="logging performance at each step"
    )
    parser.add_argument("--fix_seed", action="store_true", default=False)
    args = parser.parse_args()
    main(args)

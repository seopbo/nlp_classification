import argparse
import pickle
import torch
import torch.nn as nn
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


def main(args):
    dataset_config = Config(args.dataset_config)
    model_config = Config(args.model_config)
    ptr_config_info = Config(f"conf/pretrained/{model_config.type}.json")

    exp_dir = Path("experiments") / model_config.type
    exp_dir = exp_dir.joinpath(
        f"epochs_{args.epochs}_batch_size_{args.batch_size}_learning_rate_{args.learning_rate}"
        f"_weight_decay_{args.weight_decay}"
    )

    preprocessor = get_preprocessor(ptr_config_info, model_config)

    with open(ptr_config_info.config, mode="r") as io:
        ptr_config = json.load(io)

    # model (restore)
    checkpoint_manager = CheckpointManager(exp_dir)
    checkpoint = checkpoint_manager.load_checkpoint('best.tar')
    config = BertConfig()
    config.update(ptr_config)
    model = PairwiseClassifier(config, num_classes=model_config.num_classes, vocab=preprocessor.vocab)
    model.load_state_dict(checkpoint['model_state_dict'])

    # evaluation
    filepath = getattr(dataset_config, args.data)
    ds = Corpus(filepath, preprocessor.preprocess)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=4)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    summary_manager = SummaryManager(exp_dir)
    summary = evaluate(model, dl, {'loss': nn.CrossEntropyLoss(), 'acc': acc}, device)

    summary_manager.load('summary.json')
    summary_manager.update({'{}'.format(args.data): summary})
    summary_manager.save('summary.json')

    print('loss: {:.3f}, acc: {:.2%}'.format(summary['loss'], summary['acc']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_config", default="conf/dataset/qpair.json")
    parser.add_argument("--data", default="test")
    parser.add_argument("--model_config", default="conf/model/pairwise_skt.json")
    parser.add_argument("--epochs", default=3, help="number of epochs of training")
    parser.add_argument("--batch_size", default=64, help="batch size of training")
    parser.add_argument(
        "--learning_rate", default=1e-3, help="learning rate of training"
    )
    parser.add_argument(
        "--weight_decay", default=5e-4, help="weight decay of training"
    )
    args = parser.parse_args()
    main(args)

import argparse
import torch
import torch.nn as nn
import pickle
from pathlib import Path
from torch.utils.data import DataLoader
from model.data import Corpus
from model.net import SenCNN
from model.split import split_morphs
from model.utils import Tokenizer, PadSequence
from model.metric import evaluate, acc
from utils import Config, CheckpointManager, SummaryManager


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data', help="Directory containing config.json of data")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing config.json of model")
parser.add_argument('--dataset', default='test', help="name of the data in --data_dir to be evaluate")


if __name__ == '__main__':
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    data_config = Config(data_dir / 'config.json')
    model_config = Config(model_dir / 'config.json')

    # tokenizer
    with open(data_config.vocab, mode='rb') as io:
        vocab = pickle.load(io)
    pad_sequence = PadSequence(length=model_config.length, pad_val=vocab.to_indices(vocab.padding_token))
    tokenizer = Tokenizer(vocab=vocab, split_fn=split_morphs, pad_fn=pad_sequence)

    # model (restore)
    checkpoint_manager = CheckpointManager(model_dir)
    checkpoint = checkpoint_manager.load_checkpoint('best.tar')
    model = SenCNN(num_classes=model_config.num_classes, vocab=tokenizer.vocab)
    model.load_state_dict(checkpoint['model_state_dict'])

    # evaluation
    summary_manager = SummaryManager(model_dir)
    filepath = getattr(data_config, args.dataset)
    ds = Corpus(filepath, tokenizer.split_and_transform)
    dl = DataLoader(ds, batch_size=model_config.batch_size, num_workers=4)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    summary = evaluate(model, dl, {'loss': nn.CrossEntropyLoss(), 'acc': acc}, device)

    summary_manager.load('summary.json')
    summary_manager.update({'{}'.format(args.dataset): summary})
    summary_manager.save('summary.json')

    print('loss: {:.3f}, acc: {:.2%}'.format(summary['loss'], summary['acc']))

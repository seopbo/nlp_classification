import argparse
import torch
import torch.nn as nn
import pickle
from pathlib import Path
from model.data import Corpus, Tokenizer
from model.net import SenCNN
from model.metric import evaluate, get_accuracy
from utils import Config, CheckpointManager
from torch.utils.data import DataLoader
from mecab import MeCab
from gluonnlp.data import PadSequence


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data', help="Directory containing config.json of data")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing config.json of model")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")


if __name__ == '__main__':
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    data_config = Config(json_path=data_dir / 'config.json')
    model_config = Config(json_path=model_dir / 'config.json')

    # tokenizer
    with open(data_config.vocab, mode='rb') as io:
        vocab = pickle.load(io)
    padder = PadSequence(length=model_config.length, pad_val=vocab.to_indices(vocab.padding_token))
    tokenizer = Tokenizer(vocab=vocab, split_fn=MeCab().morphs, pad_fn=padder)

    # model (restore)
    manager = CheckpointManager(model_dir)
    ckpt = manager.load_checkpoint(args.restore_file + '.tar')
    model = SenCNN(num_classes=model_config.num_classes, vocab=tokenizer.vocab)
    model.load_state_dict(ckpt['model_state_dict'])

    # evaluation
    tst_ds = Corpus(data_config.tst, tokenizer.split_and_transform)
    tst_dl = DataLoader(tst_ds, batch_size=model_config.batch_size, num_workers=4)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    tst_summ = evaluate(model, tst_dl, {'loss': nn.CrossEntropyLoss(), 'acc': get_accuracy}, device)

    manager.load_summary('summary.json')
    manager.update_summary({'tst': tst_summ})
    manager.save_summary('summary.json')

    print('tst_loss: {:.3f}, tst_acc: {:.2%}'.format(tst_summ['loss'], tst_summ['acc']))
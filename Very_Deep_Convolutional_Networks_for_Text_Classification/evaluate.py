import pickle
import json
import fire
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from model.data import Corpus, Tokenizer
from model.net import VDCNN
from model.utils import split_to_jamo
from gluonnlp.data import PadSequence
from tqdm import tqdm


def get_accuracy(model, data_loader, device):
    if model.training:
        model.training

    correct_count = 0
    for mb in tqdm(data_loader, desc='steps'):
        x_mb, y_mb = map(lambda elm: elm.to(device), mb)

        with torch.no_grad():
            y_mb_hat = torch.max(model(x_mb), 1)[1]
            correct_count += (y_mb_hat == y_mb).sum().item()
    else:
        acc = correct_count / len(data_loader.dataset)
    return acc


def main(json_path):
    cwd = Path.cwd()
    with open(cwd / json_path) as io:
        params = json.loads(io.read())

    # tokenizer
    vocab_path = params['filepath'].get('vocab')
    with open(cwd / vocab_path, mode='rb') as io:
        vocab = pickle.load(io)
    length = params['padder'].get('length')
    padder = PadSequence(length=length, pad_val=vocab.to_indices(vocab.padding_token))
    tokenizer = Tokenizer(vocab=vocab, split_fn=split_to_jamo, pad_fn=padder)

    # model (restore)
    save_path = cwd / params['filepath'].get('ckpt')
    ckpt = torch.load(save_path)
    num_classes = params['model'].get('num_classes')
    embedding_dim = params['model'].get('embedding_dim')
    k_max = params['model'].get('k_max')

    model = VDCNN(num_classes=num_classes, embedding_dim=embedding_dim, k_max=k_max, vocab=tokenizer.vocab)
    model.load_state_dict(ckpt['model_state_dict'])

    # evaluation
    batch_size = params['training'].get('batch_size')
    tr_path = cwd / params['filepath'].get('tr')
    val_path = cwd / params['filepath'].get('val')
    tst_path = cwd / params['filepath'].get('tst')

    tr_ds = Corpus(tr_path, tokenizer.split_and_transform)
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, num_workers=4)
    val_ds = Corpus(val_path, tokenizer.split_and_transform)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=4)
    tst_ds = Corpus(tst_path, tokenizer.split_and_transform)
    tst_dl = DataLoader(tst_ds, batch_size=batch_size, num_workers=4)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    tr_acc = get_accuracy(model, tr_dl, device)
    val_acc = get_accuracy(model, val_dl, device)
    tst_acc = get_accuracy(model, tst_dl, device)

    print('tr_acc: {:.2%}, val_acc: {:.2%}, tst_acc: {:.2%}'.format(tr_acc, val_acc, tst_acc))


if __name__ == '__main__':
    fire.Fire(main)
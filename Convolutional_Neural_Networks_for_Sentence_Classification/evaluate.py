import torch
import json
import fire
import pickle
from pathlib import Path
from model.data import Corpus, Tokenizer
from model.net import SenCNN
from torch.utils.data import DataLoader
from mecab import MeCab
from gluonnlp.data import PadSequence
from tqdm import tqdm


def get_accuracy(model, data_loader, device):
    if model.training:
        model.eval()

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
    # parsing json
    proj_dir = Path.cwd()
    with open(proj_dir / json_path) as io:
        params = json.loads(io.read())

    # restoring model
    savepath = proj_dir / params['filepath'].get('ckpt')
    ckpt = torch.load(savepath)
    vocab_filepath = params['filepath'].get('vocab')

    ## common params
    with open(proj_dir / vocab_filepath, mode='rb') as io:
        vocab = pickle.load(io)

    padder = PadSequence(length=params['padder'].get('length'), pad_val=vocab.to_indices(vocab.padding_token))
    tokenizer = Tokenizer(vocab=vocab, split_fn=MeCab().morphs, pad_fn=padder)

    model = SenCNN(num_classes=params['model'].get('num_classes'), vocab=tokenizer.vocab)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # creating dataset, dataloader

    tr_filepath = proj_dir / params['filepath'].get('tr')
    val_filepath = proj_dir / params['filepath'].get('val')
    tst_filepath = proj_dir / params['filepath'].get('tst')

    tr_ds = Corpus(tr_filepath, tokenizer.split_and_transform)
    tr_dl = DataLoader(tr_ds, batch_size=128, num_workers=4)
    val_ds = Corpus(val_filepath, tokenizer.split_and_transform)
    val_dl = DataLoader(val_ds, batch_size=128, num_workers=4)
    tst_ds = Corpus(tst_filepath, tokenizer.split_and_transform)
    tst_dl = DataLoader(tst_ds, batch_size=128, num_workers=4)


    # evaluation
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    tr_acc = get_accuracy(model, tr_dl, device)
    val_acc = get_accuracy(model, val_dl, device)
    tst_acc = get_accuracy(model, tst_dl, device)

    print('tr_acc: {:.2%}, val_acc : {:.2%}, tst_acc: {:.2%}'.format(tr_acc, val_acc, tst_acc))


if __name__ == '__main__':
    fire.Fire(main)

import os
import json
import fire
import torch
from torch.utils.data import DataLoader
from model.utils import collate_fn
from model.data import Corpus
from model.net import SelfAttentiveNet
from mecab import MeCab
from tqdm import tqdm

def get_accuracy(model, dataloader, device):
    correct_count = 0
    total_count = 0
    for mb in tqdm(dataloader, desc='steps'):
        x_mb, y_mb, x_len_mb = map(lambda elm: elm.to(device), mb)

        with torch.no_grad():
            score, _ = model(x_mb, x_len_mb)
            y_mb_hat = torch.max(score, 1)[1]
            correct_count += (y_mb_hat == y_mb).sum().item()
            total_count += x_mb.size()[0]
    else:
        acc = correct_count / total_count
    return acc

def main(cfgpath):
    # parsing json
    with open(os.path.join(os.getcwd(), cfgpath)) as io:
        params = json.loads(io.read())

    # restoring model
    savepath = os.path.join(os.getcwd(), params['filepath'].get('ckpt'))
    ckpt = torch.load(savepath)

    vocab = ckpt['vocab']
    num_classes = params['model'].get('num_classes')
    lstm_hidden_dim = params['model'].get('lstm_hidden_dim')
    da = params['model'].get('da')
    r = params['model'].get('r')
    hidden_dim = params['model'].get('hidden_dim')

    model = SelfAttentiveNet(num_classes=num_classes, lstm_hidden_dim=lstm_hidden_dim,
                             da=da, r=r, hidden_dim=hidden_dim, vocab=vocab)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # creating dataset, dataloader
    tokenizer = MeCab()
    tst_filepath = os.path.join(os.getcwd(), params['filepath'].get('tst'))
    tr_filepath = os.path.join(os.getcwd(), params['filepath'].get('tr'))
    val_filepath = os.path.join(os.getcwd(), params['filepath'].get('val'))
    batch_size = params['training'].get('batch_size')

    tr_ds = Corpus(tr_filepath, tokenizer, vocab)
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, num_workers=4, collate_fn=collate_fn)
    val_ds = Corpus(val_filepath, tokenizer, vocab)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=4, collate_fn=collate_fn)
    tst_ds = Corpus(tst_filepath, tokenizer, vocab)
    tst_dl = DataLoader(tst_ds, batch_size=batch_size, num_workers=4, collate_fn=collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    tr_acc = get_accuracy(model, tr_dl, device)
    val_acc = get_accuracy(model, val_dl, device)
    tst_acc = get_accuracy(model, tst_dl, device)

    print('tr_acc: {:.2%}, val_acc: {:.2%}, tst_acc: {:.2%}'.format(tr_acc, val_acc, tst_acc))

if __name__ == '__main__':
    fire.Fire(main)

import json
import fire
import torch
import pickle
import itertools
from pathlib import Path
from torch.utils.data import DataLoader
from model.utils import batchify, split_to_self
from model.data import Corpus, Tokenizer
from model.net import BilstmCRF
from tqdm import tqdm
from sklearn.metrics import f1_score


def get_f1_score(model, data_loader, device):
    if model.training:
        model.eval()

    true_entities = []
    pred_entities = []

    for mb in tqdm(data_loader, desc='steps'):
        x_mb, y_mb, _ = map(lambda elm: elm.to(device), mb)
        y_mb = y_mb.cpu()

        with torch.no_grad():
            _, yhat = model(x_mb)
            pred_entities.extend(list(itertools.chain.from_iterable(yhat)))
            true_entities.extend(y_mb.masked_select(y_mb.ne(0)).numpy().tolist())
    else:
        score = f1_score(true_entities, pred_entities, average='weighted')
    return score


def main(json_path):
    cwd = Path.cwd()
    with open(cwd / json_path) as io:
        params = json.loads(io.read())

    # tokenizer
    token_vocab_path = params['filepath'].get('token_vocab')
    label_vocab_path = params['filepath'].get('label_vocab')
    with open(token_vocab_path, mode='rb') as io:
        token_vocab = pickle.load(io)
    with open(label_vocab_path, mode='rb') as io:
        label_vocab = pickle.load(io)
    token_tokenizer = Tokenizer(token_vocab, split_to_self)
    label_tokenizer = Tokenizer(label_vocab, split_to_self)

    # model (restore)
    save_path = cwd / params['filepath'].get('ckpt')
    ckpt = torch.load(save_path)
    lstm_hidden_dim = params['model'].get('lstm_hidden_dim')
    model = BilstmCRF(label_vocab, token_vocab, lstm_hidden_dim)
    model.load_state_dict(ckpt['model_state_dict'])

    # evaluation
    batch_size = params['training'].get('batch_size')
    tr_path = cwd / params['filepath'].get('tr')
    val_path = cwd / params['filepath'].get('val')

    tr_ds = Corpus(tr_path, token_tokenizer.split_and_transform, label_tokenizer.split_and_transform)
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, num_workers=4, collate_fn=batchify)
    val_ds = Corpus(val_path, token_tokenizer.split_and_transform, label_tokenizer.split_and_transform)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=4, collate_fn=batchify)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    tr_f1_score = get_f1_score(model, tr_dl, device)
    val_f1_score = get_f1_score(model, val_dl, device)

    print('tr_f1_score: {:.2%}, val_f1_score: {:.2%}'.format(tr_f1_score, val_f1_score))


if __name__ == '__main__':
    fire.Fire(main)

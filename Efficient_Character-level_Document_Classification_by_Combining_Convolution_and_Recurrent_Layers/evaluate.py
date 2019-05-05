import json
import fire
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from model.utils import collate_fn
from model.utils import JamoTokenizer
from model.data import Corpus
from model.net import ConvRec
from tqdm import tqdm


def get_accuracy(model, dataloader, device):
    if model.training:
        model.eval()

    correct_count = 0
    total_count = 0
    for mb in tqdm(dataloader, desc='steps'):
        x_mb, y_mb, _ = map(lambda elm: elm.to(device), mb)

        with torch.no_grad():
            y_mb_hat = torch.max(model(x_mb), 1)[1]
            correct_count += (y_mb_hat == y_mb).sum().item()
            total_count += x_mb.size()[0]
    else:
        acc = correct_count / total_count
    return acc


def main(cfgpath):
    # parsing json
    proj_dir = Path.cwd()
    with open(proj_dir / cfgpath) as io:
        params = json.loads(io.read())

    # restoring model
    savepath = proj_dir / params['filepath'].get('ckpt')
    ckpt = torch.load(savepath)
    tokenizer = JamoTokenizer()

    # creating model
    num_classes = params['model'].get('num_classes')
    embedding_dim = params['model'].get('embedding_dim')
    hidden_dim = params['model'].get('hidden_dim')

    model = ConvRec(num_classes=num_classes, embedding_dim=embedding_dim,
                    hidden_dim=hidden_dim, dic=tokenizer.token2idx)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # creating dataset, dataloader
    tst_filepath = proj_dir / params['filepath'].get('tst')
    tr_filepath = proj_dir / params['filepath'].get('tr')
    val_filepath = proj_dir / params['filepath'].get('val')
    batch_size = params['training'].get('batch_size')
    min_length = params['training'].get('min_length')

    tr_ds = Corpus(tr_filepath, tokenizer, min_length)
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, num_workers=4, collate_fn=collate_fn)
    val_ds = Corpus(val_filepath, tokenizer, min_length)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=4, collate_fn=collate_fn)
    tst_ds = Corpus(tst_filepath, tokenizer, min_length)
    tst_dl = DataLoader(tst_ds, batch_size=batch_size, num_workers=4, collate_fn=collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    tr_acc = get_accuracy(model, tr_dl, device)
    val_acc = get_accuracy(model, val_dl, device)
    tst_acc = get_accuracy(model, tst_dl, device)

    print('tr_acc: {:.2%}, val_acc: {:.2%}, tst_acc: {:.2%}'.format(tr_acc, val_acc, tst_acc))


if __name__ == '__main__':
    fire.Fire(main)
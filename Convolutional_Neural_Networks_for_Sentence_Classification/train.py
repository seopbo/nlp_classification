import pickle
import json
import fire
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from mecab import MeCab
from model.data import Corpus
from model.net import SenCNN
from gluonnlp.data import PadSequence
from tqdm import tqdm
from tensorboardX import SummaryWriter

def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    avg_loss = 0
    for step, mb in tqdm(enumerate(dataloader), desc='steps', total=len(dataloader)):
        x_mb, y_mb = map(lambda elm: elm.to(device), mb)

        with torch.no_grad():
            mb_loss = loss_fn(model(x_mb), y_mb)
        avg_loss += mb_loss.item()

    else:
        avg_loss /= (step + 1)
    return avg_loss

def main(cfgpath):
    proj_dir = Path('.')
    with open(proj_dir / cfgpath) as io:
        params = json.loads(io.read())

    tr_filepath = proj_dir / params['filepath'].get('tr')
    val_filepath = proj_dir / params['filepath'].get('val')
    vocab_filepath = params['filepath'].get('vocab')

    ## common params
    with open(proj_dir / vocab_filepath, mode='rb') as io:
        vocab = pickle.load(io)

    ## model params
    num_classes = params['model'].get('num_classes')

    ## dataset, dataloader params
    length = params['padder'].get('length')
    batch_size = params['training'].get('batch_size')
    epochs = params['training'].get('epochs')
    learning_rate = params['training'].get('learning_rate')

    # creating model
    model = SenCNN(num_classes=num_classes, vocab=vocab)

    # creating dataset, dataloader
    tokenizer = MeCab()
    padder = PadSequence(length=length)
    tr_ds = Corpus(tr_filepath, vocab, tokenizer, padder)
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_ds = Corpus(val_filepath, vocab, tokenizer, padder)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    # training
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(opt, patience=5)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    writer = SummaryWriter(log_dir='./runs/exp')
    for epoch in tqdm(range(epochs), desc='epochs'):

        tr_loss = 0

        model.train()
        for step, mb in tqdm(enumerate(tr_dl), desc='steps', total=len(tr_dl)):
            x_mb, y_mb = map(lambda elm: elm.to(device), mb)

            opt.zero_grad()
            mb_loss = loss_fn(model(x_mb), y_mb)
            mb_loss.backward()
            clip_grad_norm_(model._fc.weight, 5)
            opt.step()

            tr_loss += mb_loss.item()

            if (epoch * len(tr_dl) + step) % 300 == 0:
                val_loss = evaluate(model, val_dl, loss_fn, device)
                writer.add_scalars('loss', {'train': tr_loss / (step + 1),
                                            'validation': val_loss}, epoch * len(tr_dl) + step)
                model.train()
        else:
            tr_loss /= (step + 1)

        val_loss = evaluate(model, val_dl, loss_fn, device)
        scheduler.step(val_loss)
        tqdm.write('epoch : {}, tr_loss : {:.3f}, val_loss : {:.3f}'.format(epoch + 1, tr_loss, val_loss))

    ckpt = {'model_state_dict': model.state_dict(),
            'opt_state_dict': opt.state_dict(),
            'vocab': vocab}

    savepath = proj_dir / params['filepath'].get('ckpt')
    torch.save(ckpt, savepath)

if __name__ == '__main__':
    fire.Fire(main)
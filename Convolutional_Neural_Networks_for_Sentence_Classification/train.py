import pickle
import json
import torch
import torch.nn as nn
import torch.optim as optim
import fire
from pathlib import Path
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from mecab import MeCab
from model.data import Corpus, Tokenizer
from model.net import SenCNN
from gluonnlp.data import PadSequence
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def evaluate(model, data_loader, loss_fn, device):
    if model.training:
        model.eval()

    avg_loss = 0

    for step, mb in tqdm(enumerate(data_loader), desc='steps', total=len(data_loader)):
        x_mb, y_mb = map(lambda elm: elm.to(device), mb)

        with torch.no_grad():
            mb_loss = loss_fn(model(x_mb), y_mb)
        avg_loss += mb_loss.item()
    else:
        avg_loss /= (step + 1)

    return avg_loss


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
    tokenizer = Tokenizer(vocab=vocab, split_fn=MeCab().morphs, pad_fn=padder)

    # model
    num_classes = params['model'].get('num_classes')
    model = SenCNN(num_classes=num_classes, vocab=tokenizer.vocab)

    # training
    epochs = params['training'].get('epochs')
    batch_size = params['training'].get('batch_size')
    learning_rate = params['training'].get('learning_rate')
    global_step = params['training'].get('global_step')

    tr_path = cwd / params['filepath'].get('tr')
    val_path = cwd / params['filepath'].get('val')
    tr_ds = Corpus(tr_path, tokenizer.split_and_transform)
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_ds = Corpus(val_path, tokenizer.split_and_transform)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(params=model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(opt, patience=5)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    writer = SummaryWriter('./runs/{}'.format(params['version']))
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

            if (epoch * len(tr_dl) + step) % global_step == 0:
                val_loss = evaluate(model, val_dl, loss_fn, device)
                writer.add_scalars('loss', {'train': tr_loss / (step + 1),
                                            'val': val_loss}, epoch * len(tr_dl) + step)

                model.train()
        else:
            tr_loss /= (step + 1)

        val_loss = evaluate(model, val_dl, loss_fn, device)
        scheduler.step(val_loss)
        tqdm.write('epoch : {}, tr_loss : {:.3f}, val_loss : {:.3f}'.format(epoch + 1, tr_loss, val_loss))

    ckpt = {'model_state_dict': model.state_dict(),
            'opt_state_dict': opt.state_dict()}

    save_path = cwd / params['filepath'].get('ckpt')
    torch.save(ckpt, save_path)


if __name__ == '__main__':
    fire.Fire(main)
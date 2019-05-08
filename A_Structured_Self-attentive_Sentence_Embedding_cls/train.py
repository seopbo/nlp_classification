import json
import pickle
import fire
import torch
import torch.nn as nn
from pathlib import Path
from torch import optim
from torch.optim.lr_scheduler import  ReduceLROnPlateau
from torch.utils.data import DataLoader
from mecab import MeCab
from model.utils import collate_fn
from model.data import Corpus
from model.net import SAN
from tqdm import tqdm
from tensorboardX import SummaryWriter


def evaluate(model, dataloader, loss_fn, device):
    if model.training:
        model.eval()

    model.eval()
    avg_loss = 0
    for step, mb in tqdm(enumerate(dataloader), desc='steps', total=len(dataloader)):
        x_mb, y_mb, _ = map(lambda elm: elm.to(device), mb)

        with torch.no_grad():
            score, _ = model(x_mb)
            mb_loss = loss_fn(score, y_mb)
        avg_loss += mb_loss.item()
    else:
        avg_loss /= (step + 1)

    return avg_loss


def regularize(attn_mat, r, device):
    with torch.no_grad():
        sim_mat = torch.bmm(attn_mat, attn_mat.permute(0, 2, 1))
        identity = torch.eye(r).to(device)
        p = torch.norm(sim_mat - identity, dim=(1, 2)).mean()
    return p


def main(cfgpath, global_step):
    # parsing json
    proj_dir = Path.cwd()
    with open(proj_dir / cfgpath) as io:
        params = json.loads(io.read())

    tr_filepath = proj_dir / params['filepath'].get('tr')
    val_filepath = proj_dir / params['filepath'].get('val')
    vocab_filepath = params['filepath'].get('vocab')

    ## common params
    tokenizer = MeCab()
    with open(vocab_filepath, mode='rb') as io:
        vocab = pickle.load(io)

    ## model params
    num_classes = params['model'].get('num_classes')
    lstm_hidden_dim = params['model'].get('lstm_hidden_dim')
    da = params['model'].get('da')
    r = params['model'].get('r')
    hidden_dim = params['model'].get('hidden_dim')

    ## dataset, dataloader params
    batch_size = params['training'].get('batch_size')
    epochs = params['training'].get('epochs')
    learning_rate = params['training'].get('learning_rate')

    # creating model
    model = SAN(num_classes=num_classes, lstm_hidden_dim=lstm_hidden_dim,
                da=da, r=r, hidden_dim=hidden_dim, vocab=vocab)

    # creating dataset, dataloader
    tr_ds = Corpus(tr_filepath, tokenizer, vocab)
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True,
                       collate_fn=collate_fn)
    val_ds = Corpus(val_filepath, tokenizer, vocab)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=4,
                        collate_fn=collate_fn)

    # training
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(params=model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(opt, patience=5)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)
    writer = SummaryWriter(log_dir='./runs/exp')

    for epoch in tqdm(range(epochs), desc='epochs'):

        tr_loss = 0

        model.train()
        for step, mb in tqdm(enumerate(tr_dl), desc='steps', total=len(tr_dl)):
            x_mb, y_mb, _ = map(lambda elm: elm.to(device), mb)

            opt.zero_grad()
            score, attn_mat = model(x_mb)
            reg = regularize(attn_mat, r, device)
            mb_loss = loss_fn(score, y_mb)
            mb_loss.add_(reg)
            mb_loss.backward()
            opt.step()

            tr_loss += mb_loss.item()

            if (epoch * len(tr_dl) + step) % global_step == 0:
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
            'opt_state_dict': opt.state_dict()}

    savepath = proj_dir / params['filepath'].get('ckpt')
    torch.save(ckpt, savepath)


if __name__ == '__main__':
    fire.Fire(main)
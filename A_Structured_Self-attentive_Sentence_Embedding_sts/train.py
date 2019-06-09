import json
import pickle
import fire
import torch
import torch.nn as nn
from pathlib import Path
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from mecab import MeCab
from model.data import Corpus, Tokenizer
from model.utils import batchify
from model.net import SAN
from tqdm import tqdm


def evaluate(model, data_loader, loss_fn, device):
    if model.training:
        model.eval()

    avg_loss = 0
    for step, mb in tqdm(enumerate(data_loader), desc='steps', total=len(data_loader)):
        queries_a_mb, queries_b_mb, y_mb = map(lambda elm: elm.to(device), mb)
        queries_mb = (queries_a_mb, queries_b_mb)

        with torch.no_grad():
            score, _, _ = model(queries_mb)
            mb_loss = loss_fn(score, y_mb)

        avg_loss += mb_loss.item()
    else:
        avg_loss /= (step + 1)

    return avg_loss


def regularize(attn_mat, r, device):
    sim_mat = torch.bmm(attn_mat, attn_mat.permute(0, 2, 1))
    identity = torch.eye(r).to(device)
    p = torch.norm(sim_mat - identity, dim=(1, 2)).mean()
    return p


def main(json_path):
    cwd = Path.cwd()
    with open(cwd / json_path) as io:
        params = json.loads(io.read())

    # tokenizer
    vocab_path = params['filepath'].get('vocab')
    with open(cwd / vocab_path, mode='rb') as io:
        vocab = pickle.load(io)
    tokenizer = Tokenizer(vocab=vocab, split_fn=MeCab().morphs)

    # model
    num_classes = params['model'].get('num_classes')
    lstm_hidden_dim = params['model'].get('lstm_hidden_dim')
    hidden_dim = params['model'].get('hidden_dim')
    da = params['model'].get('da')
    r = params['model'].get('r')
    model = SAN(num_classes=num_classes, lstm_hidden_dim=lstm_hidden_dim, hidden_dim=hidden_dim,
                da=da, r=r, vocab=tokenizer.vocab)

    # training
    epochs = params['training'].get('epochs')
    batch_size = params['training'].get('batch_size')
    learning_rate = params['training'].get('learning_rate')
    global_step = params['training'].get('global_step')

    tr_path = cwd / params['filepath'].get('tr')
    val_path = cwd / params['filepath'].get('val')
    tr_ds = Corpus(tr_path, tokenizer.split_and_transform)
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, collate_fn=batchify)
    val_ds = Corpus(val_path, tokenizer.split_and_transform)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=4, collate_fn=batchify)

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
            queries_a_mb, queries_b_mb, y_mb = map(lambda elm: elm.to(device), mb)
            queries_mb = (queries_a_mb, queries_b_mb)

            opt.zero_grad()
            score, queries_a_attn_mat, queries_b_attn_mat = model(queries_mb)
            a_reg = regularize(queries_a_attn_mat, r, device)
            b_reg = regularize(queries_b_attn_mat, r, device)
            mb_loss = loss_fn(score, y_mb)
            mb_loss.add_(a_reg)
            mb_loss.add_(b_reg)
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

    save_path = cwd / params['filepath'].get('ckpt')
    torch.save(ckpt, save_path)


if __name__ == '__main__':
    fire.Fire(main)
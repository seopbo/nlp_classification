import json
import pickle
import fire
import torch
from pathlib import Path
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model.net import BilstmCRF
from model.data import Corpus, Tokenizer
from model.utils import batchify, split_to_self
from tqdm import tqdm


def evaluate(model, data_loader, device):
    if model.training:
        model.eval()

    model.eval()
    avg_loss = 0
    for step, mb in tqdm(enumerate(data_loader), desc='steps', total=len(data_loader)):
        x_mb, y_mb, _ = map(lambda elm: elm.to(device), mb)

        with torch.no_grad():
            mb_loss = model.loss(x_mb, y_mb)
        avg_loss += mb_loss.item()
    else:
        avg_loss /= (step + 1)

    return avg_loss


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

    # model
    lstm_hidden_dim = params['model'].get('lstm_hidden_dim')
    model = BilstmCRF(label_vocab, token_vocab, lstm_hidden_dim)

    # training
    epochs = params['training'].get('epochs')
    batch_size = params['training'].get('batch_size')
    learning_rate = params['training'].get('learning_rate')
    global_step = params['training'].get('global_step')

    # creating dataset, dataloader
    tr_path = cwd / params['filepath'].get('tr')
    val_path = cwd / params['filepath'].get('val')
    tr_ds = Corpus(tr_path, token_tokenizer.split_and_transform, label_tokenizer.split_and_transform)
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True,
                       collate_fn=batchify)
    val_ds = Corpus(val_path, token_tokenizer.split_and_transform, label_tokenizer.split_and_transform)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=4, collate_fn=batchify)

    # training
    opt = optim.Adam(params=model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(opt, patience=5)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    writer = SummaryWriter('./runs/{}'.format(params['version']))
    for epoch in tqdm(range(epochs), desc='epochs'):

        tr_loss = 0

        model.train()
        for step, mb in tqdm(enumerate(tr_dl), desc='steps', total=len(tr_dl)):
            x_mb, y_mb, _ = map(lambda elm: elm.to(device), mb)

            opt.zero_grad()
            mb_loss = model.loss(x_mb, y_mb)
            mb_loss.backward()
            opt.step()

            tr_loss += mb_loss.item()

            if (epoch * len(tr_dl) + step) % global_step == 0:
                val_loss = evaluate(model, val_dl, device)
                writer.add_scalars('loss', {'train': tr_loss / (step + 1),
                                            'validation': val_loss}, epoch * len(tr_dl) + step)
                model.train()
        else:
            tr_loss /= (step + 1)

        val_loss = evaluate(model, val_dl, device)
        scheduler.step(val_loss)
        tqdm.write('epoch : {}, tr_loss : {:.3f}, val_loss : {:.3f}'.format(epoch + 1, tr_loss, val_loss))

    ckpt = {'model_state_dict': model.state_dict(),
            'opt_state_dict': opt.state_dict()}

    save_path = cwd / params['filepath'].get('ckpt')
    torch.save(ckpt, save_path)


if __name__ == '__main__':
    fire.Fire(main)
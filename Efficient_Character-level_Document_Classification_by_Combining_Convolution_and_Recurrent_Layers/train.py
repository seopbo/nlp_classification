import os
import json
import fire
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from model.utils import collate_fn
from model.utils import JamoTokenizer
from model.data import Corpus
from model.net import ConvRec
from tqdm import tqdm
from tensorboardX import SummaryWriter

def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    avg_loss = 0
    for step, mb in tqdm(enumerate(dataloader), desc='steps', total=len(dataloader)):
        x_mb, y_mb, x_len_mb = map(lambda elm: elm.to(device), mb)

        with torch.no_grad():
            mb_loss = loss_fn(model(x_mb, x_len_mb), y_mb)
        avg_loss += mb_loss.item()
    else:
        avg_loss /= (step + 1)

    return avg_loss

def main(cfgpath):
    # parsing json
    with open(os.path.join(os.getcwd(), cfgpath)) as io:
        params = json.loads(io.read())

    tr_filepath = os.path.join(os.getcwd(), params['filepath'].get('tr'))
    val_filepath = os.path.join(os.getcwd(), params['filepath'].get('val'))

    ## common params
    tokenizer = JamoTokenizer()

    ## model params
    num_classes = params['model'].get('num_classes')
    embedding_dim = params['model'].get('embedding_dim')
    hidden_dim = params['model'].get('hidden_dim')

    ## dataset, dataloader params
    min_length = params['training'].get('min_length')
    batch_size = params['training'].get('batch_size')
    epochs = params['training'].get('epochs')
    learning_rate = params['training'].get('learning_rate')

    # creating model
    model = ConvRec(num_classes=num_classes, embedding_dim=embedding_dim,
                    hidden_dim=hidden_dim, dic=tokenizer.token2idx)

    # creating dataset, dataloader
    tr_ds = Corpus(tr_filepath, tokenizer, min_length)
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True,
                       collate_fn=collate_fn)
    val_ds = Corpus(val_filepath, tokenizer, min_length)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=4,
                        collate_fn=collate_fn)

    # training
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=5 * 10**-4)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    model.to(device)
    writer = SummaryWriter(log_dir='./runs/exp')

    for epoch in tqdm(range(epochs), desc='epochs'):

        tr_loss = 0

        model.train()
        for step, mb in tqdm(enumerate(tr_dl), desc='steps', total=len(tr_dl)):
            x_mb, y_mb, x_len_mb = map(lambda elm: elm.to(device), mb)

            opt.zero_grad()
            mb_loss = loss_fn(model(x_mb, x_len_mb), y_mb)
            mb_loss.backward()
            clip_grad_norm_(model.parameters(), 5)
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
        tqdm.write('epoch : {}, tr_loss : {:.3f}, val_loss : {:.3f}'.format(epoch + 1, tr_loss, val_loss))

    ckpt = {'model_state_dict': model.state_dict(),
            'opt_state_dict': opt.state_dict()}

    savepath = os.path.join(os.getcwd(), params['filepath'].get('ckpt'))
    torch.save(ckpt, savepath)

if __name__ == '__main__':
    fire.Fire(main)
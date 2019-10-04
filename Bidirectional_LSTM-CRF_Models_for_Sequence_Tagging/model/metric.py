import torch
import itertools
from tqdm import tqdm
from sklearn.metrics import f1_score


def evaluate(model, data_loader, device):
    if model.training:
        model.eval()

    model.eval()
    avg_loss = 0
    for step, mb in tqdm(enumerate(data_loader), desc='steps', total=len(data_loader)):
        x_mb, y_mb = map(lambda elm: elm.to(device), mb)

        with torch.no_grad():
            mb_loss = model.loss(x_mb, y_mb)
        avg_loss += mb_loss.item()
    else:
        avg_loss /= (step + 1)

    return avg_loss


def get_f1_score(model, data_loader, device):
    if model.training:
        model.eval()

    true_entities = []
    pred_entities = []

    for mb in tqdm(data_loader, desc='steps'):
        x_mb, y_mb = map(lambda elm: elm.to(device), mb)
        y_mb = y_mb.cpu()

        with torch.no_grad():
            _, yhat = model(x_mb)
            pred_entities.extend(list(itertools.chain.from_iterable(yhat)))
            true_entities.extend(y_mb.masked_select(y_mb.ne(0)).numpy().tolist())
    else:
        score = f1_score(true_entities, pred_entities, average='weighted')
    return score

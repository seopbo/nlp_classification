import torch
from tqdm import tqdm


def evaluate(model, data_loader, metrics, device):
    if model.training:
        model.eval()

    summary = {metric: 0 for metric in metrics}

    for step, mb in tqdm(enumerate(data_loader), desc='steps', total=len(data_loader)):
        x_mb, y_mb = map(lambda elm: elm.to(device), mb)

        with torch.no_grad():
            y_hat_mb = model(x_mb)

            for metric in metrics:
                summary[metric] += metrics[metric](y_hat_mb, y_mb).item()
    else:
        for metric in metrics:
            summary[metric] /= (step + 1)

    return summary


def get_accuracy(yhat, y):
    with torch.no_grad():
        yhat = yhat.max(dim=1)[1]
        acc = (yhat == y).float().mean()
    return acc
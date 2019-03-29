import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(data: list) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """custom collate_fn for DataLoader

    Args:
        data (list): list of torch.Tensors

    Returns:
        data (tuple): tuple of torch.Tensors
    """
    data = sorted(data, key=lambda elm: elm[2], reverse=True)
    indices, labels, lengths = zip(*data)
    indices = pad_sequence(indices, batch_first=True)
    labels = torch.stack(labels, 0)
    lengths = torch.stack(lengths, 0)
    return indices, labels, lengths
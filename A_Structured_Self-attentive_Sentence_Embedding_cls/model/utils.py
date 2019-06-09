import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple


def batchify(data: List[Tuple[torch.tensor, torch.tensor, torch.tensor]]) ->\
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """custom collate_fn for DataLoader

    Args:
        data (list): list of torch.Tensors

    Returns:
        data (tuple): tuple of torch.Tensors
    """
    indices, labels, lengths = zip(*data)
    indices = pad_sequence(indices, batch_first=True, padding_value=1)
    labels = torch.stack(labels, 0)
    lengths = torch.stack(lengths, 0)
    return indices, labels, lengths
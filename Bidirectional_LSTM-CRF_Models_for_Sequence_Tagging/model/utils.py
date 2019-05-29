import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple


def collate_fn(data: List[Tuple[torch.tensor, torch.tensor, torch.tensor]]) ->\
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """custom collate_fn for DataLoader

    Args:
        data (list): list of torch.Tensors

    Returns:
        data (tuple): tuple of torch.Tensors
    """
    tokens2indices, labels2indices, lengths = zip(*data)
    tokens2indices = pad_sequence(tokens2indices, batch_first=True, padding_value=1)
    labels2indices = pad_sequence(labels2indices, batch_first=True, padding_value=0)
    lengths = torch.stack(lengths, 0)
    return tokens2indices, labels2indices, lengths
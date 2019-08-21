import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple


def batchify(data: List[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor]]) \
        -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

    data = list(zip(*data))
    queries_a, queries_b, is_duplicate = data

    queries_a, _ = zip(*queries_a)
    queries_a = pad_sequence(queries_a, batch_first=True, padding_value=1)

    queries_b, _ = zip(*queries_b)
    queries_b = pad_sequence(queries_b, batch_first=True, padding_value=1)
    is_duplicate = torch.stack(is_duplicate, 0)

    return queries_a, queries_b, is_duplicate

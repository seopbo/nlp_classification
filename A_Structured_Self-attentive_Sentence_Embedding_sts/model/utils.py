import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple


def collate_fn(data: List[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor]]) \
        -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

    data = list(zip(*data))
    queries_a, queries_b, is_duplicate = data

    queries_a = sorted(queries_a, key=lambda elm: elm[1], reverse=True)
    queries_a, queries_a_len = zip(*queries_a)
    queries_a = pad_sequence(queries_a, batch_first=True, padding_value=1)
    # queries_a_len = torch.stack(queries_a_len, 0)

    queries_b = sorted(queries_b, key=lambda elm: elm[1], reverse=True)
    queries_b, queries_b_len = zip(*queries_b)
    queries_b = pad_sequence(queries_b, batch_first=True, padding_value=1)
    # queries_b_len = torch.stack(queries_b_len, 0)

    is_duplicate = torch.stack(is_duplicate, 0)

    return queries_a, queries_b, is_duplicate
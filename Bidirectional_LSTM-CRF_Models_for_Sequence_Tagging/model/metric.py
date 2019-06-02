import torch
import torch.nn.functional as F


def sequence_loss(logit, target):
    masking_mb = target.ne(0).float()
    valid_length_mb = masking_mb.sum(-1)
    log_softmax_mb = F.log_softmax(logit, dim=-1) * -1
    log_softmax_mb = torch.gather(log_softmax_mb, -1, target.unsqueeze(-1)).squeeze() * masking_mb
    ce_loss_mb = log_softmax_mb.sum(dim=1) / valid_length_mb
    ce_loss = ce_loss_mb.mean()
    return ce_loss
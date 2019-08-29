import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Union


def sequence_mask(lengths: torch.Tensor, max_len: Union[torch.Tensor, int]):
    if lengths.ndimension() != 1:
        raise ValueError

    mask = torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]
    return mask


def mask_nll_loss(logits, target, mask):
    cross_entropy = -torch.gather(F.log_softmax(logits, dim=-1), -1, target)
    cross_entropy = cross_entropy.masked_select(mask).mean()
    return cross_entropy


def evaluate(encoder, decoder, tgt_vocab, data_loader, device):
    if encoder.training:
        encoder.eval()
    if decoder.training:
        decoder.eval()

    loss = 0

    for step, mb in tqdm(enumerate(data_loader), desc='steps', total=len(data_loader)):
        mb_loss = 0
        src_mb, tgt_mb = map(lambda elm: elm.to(device), mb)

        with torch.no_grad():
            # encoder
            enc_outputs_mb, src_length_mb, enc_hc_mb = encoder(src_mb)

            # decoder
            dec_input_mb = torch.ones((tgt_mb.size()[0], 1), device=device).long()
            dec_input_mb *= tgt_vocab.to_indices(tgt_vocab.bos_token)
            dec_hc_mb = enc_hc_mb
            tgt_length_mb = tgt_mb.ne(tgt_vocab.to_indices(tgt_vocab.padding_token)).sum(dim=1)
            tgt_mask_mb = sequence_mask(tgt_length_mb, tgt_length_mb.max())

            for t in range(tgt_length_mb.max()):
                dec_output_mb, dec_hc_mb = decoder(dec_input_mb, dec_hc_mb, enc_outputs_mb, src_length_mb)
                sequence_loss = mask_nll_loss(dec_output_mb, tgt_mb[:, [t]], tgt_mask_mb[:, [t]])
                mb_loss += sequence_loss
                dec_input_mb = tgt_mb[:, [t]]  # next input is current target
            else:
                mb_loss /= tgt_length_mb.max()
                mb_loss *= src_mb.shape[0]

            loss += mb_loss.item()
    else:
        loss /= len(data_loader.dataset)

    return loss
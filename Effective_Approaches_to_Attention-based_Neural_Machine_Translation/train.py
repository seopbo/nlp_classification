import argparse
import pickle
import torch
import random
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from model.split import Stemmer
from model.net import Encoder, AttnDecoder
from model.data import NMTCorpus, batchify
from model.utils import SourceProcessor, TargetProcessor
from model.metric import mask_nll_loss, sequence_mask, evaluate
from utils import Config, CheckpointManager, SummaryManager


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data', help="Directory containing config.json of data")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing config.json of model")

args = argparse.Namespace(data_dir='data', model_dir='experiments/base_model')


if __name__ == '__main__':
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    data_config = Config(json_path=data_dir / 'config.json')
    model_config = Config(json_path=model_dir / 'config.json')
    # processor
    with open(data_config.source_vocab, mode='rb') as io:
        src_vocab = pickle.load(io)
    ko_stemmer = Stemmer(language='ko')
    src_processor = SourceProcessor(src_vocab, ko_stemmer.extract_stem)

    with open(data_config.target_vocab, mode='rb') as io:
        tgt_vocab = pickle.load(io)
    en_stemmer = Stemmer(language='en')
    tgt_processor = TargetProcessor(tgt_vocab, en_stemmer.extract_stem)

    # model
    encoder = Encoder(src_vocab, model_config.encoder_hidden_dim, model_config.drop_ratio)
    decoder = AttnDecoder(tgt_vocab, model_config.method, model_config.encoder_hidden_dim,
                          model_config.decoder_hidden_dim, model_config.drop_ratio)

    # training
    tr_ds = NMTCorpus(data_config.train, src_processor.process, tgt_processor.process)
    tr_dl = DataLoader(tr_ds, model_config.batch_size, shuffle=True, num_workers=4, collate_fn=batchify,
                       drop_last=True)
    val_ds = NMTCorpus(data_config.dev, src_processor.process, tgt_processor.process)
    val_dl = DataLoader(val_ds, model_config.batch_size, shuffle=False, num_workers=4, collate_fn=batchify,
                        drop_last=False)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    encoder.to(device)
    decoder.to(device)

    writer = SummaryWriter('{}/runs'.format(model_dir))
    checkpoint_manager = CheckpointManager(model_dir)
    summary_manager = SummaryManager(model_dir)
    best_val_loss = 1e+10

    opt = optim.Adam([{'params': encoder.parameters()},
                      {'params': decoder.parameters()}], lr=model_config.learning_rate, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(opt, patience=5, min_lr=5e-4)

    for epoch in tqdm(range(model_config.epochs), desc='epochs'):
        tr_loss = 0

        encoder.train()
        decoder.train()

        for step, mb in tqdm(enumerate(tr_dl), desc='steps', total=len(tr_dl)):
            mb_loss = 0
            src_mb, tgt_mb = map(lambda elm: elm.to(device), mb)

            opt.zero_grad()
            # encoder
            enc_outputs_mb, src_length_mb, enc_hc_mb = encoder(src_mb)

            # decoder
            use_teacher_forcing = True if random.random() > model_config.teacher_forcing_ratio else False

            dec_input_mb = torch.ones((tgt_mb.size()[0], 1), device=device).long()
            dec_input_mb *= tgt_vocab.to_indices(tgt_vocab.bos_token)
            dec_hc_mb = enc_hc_mb

            tgt_length_mb = tgt_mb.ne(tgt_vocab.to_indices(tgt_vocab.padding_token)).sum(dim=1)
            tgt_mask_mb = sequence_mask(tgt_length_mb, tgt_length_mb.max())

            if use_teacher_forcing:
                for t in range(tgt_length_mb.max()):
                    dec_output_mb, dec_hc_mb = decoder(dec_input_mb, dec_hc_mb, enc_outputs_mb, src_length_mb)
                    sequence_loss = mask_nll_loss(dec_output_mb, tgt_mb[:, [t]], tgt_mask_mb[:, [t]])
                    mb_loss += sequence_loss
                    dec_input_mb = tgt_mb[:, [t]]  # next input is current target
                else:
                    mb_loss /= tgt_length_mb.max()
            else:
                for t in range(tgt_length_mb.max()):
                    dec_output_mb, dec_hc_mb = decoder(dec_input_mb, dec_hc_mb, enc_outputs_mb, src_length_mb)
                    sequence_loss = mask_nll_loss(dec_output_mb, tgt_mb[:, [t]], tgt_mask_mb[:, [t]])
                    mb_loss += sequence_loss
                    dec_input_mb = dec_output_mb.topk(1).indices
                else:
                    mb_loss /= tgt_length_mb.max()

            # update params
            mb_loss.backward()
            nn.utils.clip_grad_norm_(encoder.parameters(), model_config.clip_norm)
            nn.utils.clip_grad_norm_(decoder.parameters(), model_config.clip_norm)
            opt.step()

            tr_loss += mb_loss.item()

            if (epoch * len(tr_dl) + step) % model_config.summary_step == 0:
                val_loss = evaluate(encoder, decoder, tgt_vocab, val_dl, device)
                writer.add_scalars('loss', {'train': tr_loss / (step + 1),
                                            'val': val_loss}, epoch * len(tr_dl) + step)
                encoder.train()
                decoder.train()

        else:
            tr_loss /= (step + 1)

            tr_summary = {'loss': tr_loss}
            val_loss = evaluate(encoder, decoder, tgt_vocab, val_dl, device)
            scheduler.step(val_loss)
            val_summary = {'loss': val_loss}
            tqdm.write('epoch : {}, tr_loss: {:.3f}, val_loss: '
                       '{:.3f}'.format(epoch + 1, tr_summary['loss'], val_summary['loss']))

            is_best = val_loss < best_val_loss

            if is_best:
                state = {'epoch': epoch + 1,
                         'encoder_state_dict': encoder.state_dict(),
                         'decoder_state_dict': decoder.state_dict(),
                         'opt_state_dict': opt.state_dict()}
                summary = {'train': tr_summary, 'validation': val_summary}

                summary_manager.update(summary)
                summary_manager.save('summary.json')
                checkpoint_manager.save_checkpoint(state, 'best.tar')

                best_val_loss = val_loss

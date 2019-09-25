import argparse
import pickle
import torch
import random
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from model.split import Stemmer
from model.net import BidiEncoder, AttnDecoder
from model.data import NMTCorpus, batchify
from model.utils import SourceProcessor, TargetProcessor
from model.metric import mask_nll_loss, sequence_mask, evaluate
from utils import Config, CheckpointManager, SummaryManager


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data', help="Directory containing config.json of data")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing config.json of model")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")
parser.add_argument('--data_name', default='test', help="name of the data in --data_dir to be evaluate")


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

    # model (restore)
    encoder = BidiEncoder(src_vocab, model_config.encoder_hidden_dim, model_config.drop_ratio)
    decoder = AttnDecoder(tgt_vocab, model_config.method, model_config.encoder_hidden_dim,
                          model_config.decoder_hidden_dim, model_config.drop_ratio)

    checkpoint_manager = CheckpointManager(model_dir)
    checkpoint = checkpoint_manager.load_checkpoint(args.restore_file + '.tar')
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    encoder.eval()
    decoder.eval()

    # evaluation
    summary_manager = SummaryManager(model_dir)
    filepath = getattr(data_config, args.data_name)
    ds = NMTCorpus(filepath, src_processor.process, tgt_processor.process)
    dl = DataLoader(ds, model_config.batch_size, shuffle=False, num_workers=4, collate_fn=batchify,
                    drop_last=False)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    encoder.to(device)
    decoder.to(device)

    loss = evaluate(encoder, decoder, tgt_vocab, dl, device)
    summary = {'loss': loss}
    summary_manager.load('summary.json')
    summary_manager.update({'{}'.format(args.data_name): summary})
    summary_manager.save('summary.json')

    print('loss: {:.3f}'.format(loss))
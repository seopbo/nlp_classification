# todo train.py 구현
import pickle
import torch
from torch.utils.data import DataLoader
from model.net import Encoder, AttnDecoder
from model.utils import SourceProcessor, TargetProcessor
from model.data import batchify, NMTCorpus
from model.split import split_morphs, split_space

with open('data/vocab_ko.pkl', mode='rb') as io:
    vocab_ko = pickle.load(io)
ko_processor = SourceProcessor(vocab_ko, split_morphs)
with open('data/vocab_en.pkl', mode='rb') as io:
    vocab_en = pickle.load(io)
en_processor = TargetProcessor(vocab_en, split_space)

ds = NMTCorpus('data/train.txt', ko_processor.process, en_processor.process)
dl = DataLoader(ds, 2, shuffle=False, num_workers=4, collate_fn=batchify)
x, y = next(iter(dl))

encoder = Encoder(vocab=ko_processor.vocab, encoder_hidden_dim=128, drop_ratio=.2)
encoder_outputs, encoder_hc, source_length = encoder(x)

decoder = AttnDecoder(vocab=en_processor.vocab, method='general', encoder_hidden_dim=128,
                      decoder_hidden_dim=128)
decoder_input = torch.LongTensor([vocab_en.to_indices(vocab_en.bos_token) for _ in range(2)]).reshape(-1,1)
decoder_output, hc = decoder(decoder_input, encoder_hc, encoder_outputs, source_length)
decoder_input = decoder_output.max(dim=1).indices.reshape(2, -1)

import argparse
import torch
import pickle
from collections import OrderedDict
from pathlib import Path
from model.utils import Vocab
from utils import Config
from pretrained.tokenization import BertTokenizer as ETRITokenizer
from gluonnlp.vocab import BERTVocab
from urllib.request import urlretrieve

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(description="prepare pretrained-bert")
parser.add_argument("--type", type=str, choices=["skt", "etri"], default="skt", required=True)


if __name__ == '__main__':
    args = parser.parse_args()
    ptr_dir = Path('pretrained')

    if args.type == 'skt':
        ptr_config_path = ptr_dir / 'bert_config_skt.json'
        ptr_bert_path = ptr_dir / 'pytorch_model_skt.bin'
        ptr_vocab_path = ptr_dir / 'pytorch_model_skt_vocab.json'
        ptr_tokenizer_path = ptr_dir /'pytorch_model_skt_tokenizer.model'

        if not ptr_bert_path.exists():
            urlretrieve('https://kobert.blob.core.windows.net/models/kobert/pytorch/pytorch_kobert_2439f391a6.params',
                        filename=ptr_bert_path)
            ptr_bert = torch.load(ptr_bert_path)
            ptr_bert = OrderedDict([(('bert.' + k), ptr_bert.get(k)) for k in ptr_bert.keys()])
            torch.save(ptr_bert, ptr_bert_path)
        else:
            print('Already you have pytorch_model_skt.bin!')

        if not ptr_vocab_path.exists():
            urlretrieve('https://kobert.blob.core.windows.net/models/kobert/vocab/kobertvocab_f38b8a4d6d.json',
                        filename=ptr_vocab_path)
            ptr_bert_vocab = BERTVocab.from_json(ptr_vocab_path.open(mode='rt').read())
            vocab = Vocab(ptr_bert_vocab.idx_to_token,
                          padding_token="[PAD]",
                          unknown_token="[UNK]",
                          bos_token=None,
                          eos_token=None,
                          reserved_tokens=["[CLS]", "[SEP]", "[MASK]"],
                          token_to_idx=ptr_bert_vocab.token_to_idx)

            # save vocab
            with open(ptr_vocab_path.with_suffix('.pkl'), mode="wb") as io:
                pickle.dump(vocab, io)
        else:
            print('Already you have pytorch_model_skt_vocab.json!')

        if not ptr_tokenizer_path.exists():
            urlretrieve('https://kobert.blob.core.windows.net/models/kobert/tokenizer/tokenizer_78b3253a26.model',
                        filename=ptr_tokenizer_path)
        else:
            print('Already you have pytorch_model_skt_tokenizer.model')

        ptr_config = Config({'config': str(ptr_config_path),
                             'bert': str(ptr_bert_path),
                             'tokenizer': str(ptr_tokenizer_path),
                             'vocab': str(ptr_vocab_path.with_suffix('.pkl'))})
        ptr_config.save(ptr_dir / "config_skt.json")

    if args.type == 'etri':
        # loading BertTokenizer
        ptr_config_path = ptr_dir / 'bert_config_etri.json'
        ptr_tokenizer_path = ptr_dir / "vocab.korean.rawtext.list"
        ptr_bert_path = ptr_dir / "pytorch_model_etri.bin"

        ptr_tokenizer = ETRITokenizer.from_pretrained(
            ptr_tokenizer_path, do_lower_case=False
        )
        # generate vocab
        idx_to_token = list(ptr_tokenizer.vocab.keys())
        token_to_idx = {token: idx for idx, token in enumerate(idx_to_token)}

        vocab = Vocab(
            idx_to_token,
            padding_token="[PAD]",
            unknown_token="[UNK]",
            bos_token=None,
            eos_token=None,
            reserved_tokens=["[CLS]", "[SEP]", "[MASK]"],
            token_to_idx=token_to_idx,
        )

        # save vocab
        ptr_vocab_path = ptr_dir / "vocab_etri.pkl"
        with open(ptr_vocab_path, mode="wb") as io:
            pickle.dump(vocab, io)

        ptr_config = Config({'config': str(ptr_config_path),
                             'bert': str(ptr_bert_path),
                             'tokenizer': str(ptr_tokenizer_path),
                             'vocab': str(ptr_vocab_path)})
        ptr_config.save(ptr_dir / "config_etri.json")




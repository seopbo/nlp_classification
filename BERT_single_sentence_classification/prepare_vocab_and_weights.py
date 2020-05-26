import argparse
import zipfile
import gdown
import torch
import pickle
from collections import OrderedDict
from pathlib import Path
from model.utils import Vocab
from utils import Config
from gluonnlp.vocab import BERTVocab
from urllib.request import urlretrieve

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(description="prepare pretrained-bert")
parser.add_argument("--type", type=str, choices=["skt", "etri"], default="skt", required=True)


if __name__ == '__main__':
    args = parser.parse_args()
    ptr_dir = Path('pretrained') / args.type

    if args.type == 'skt':
        if not ptr_dir.exists():
            ptr_dir.mkdir(parents=True)

        ptr_config_path = ptr_dir / 'bert_config_skt.json'
        ptr_bert_path = ptr_dir / 'pytorch_model_skt.bin'
        ptr_vocab_path = ptr_dir / 'pytorch_model_skt_vocab.json'
        ptr_tokenizer_path = ptr_dir /'pytorch_model_skt_tokenizer.model'

        if not ptr_config_path.exists():
            url = "https://drive.google.com/uc?id=1AmNxscnp_sQyN6nGKwkvXVzv0Bzafcpo"
            gdown.download(url, output=str(ptr_config_path))
            print("Processing is done")
        else:
            print("Already you have bert_config_skt.json")

        if not ptr_bert_path.exists():
            urlretrieve('https://kobert.blob.core.windows.net/models/kobert/pytorch/pytorch_kobert_2439f391a6.params',
                        filename=ptr_bert_path)
            ptr_bert = torch.load(ptr_bert_path)
            ptr_bert = OrderedDict([(('bert.' + k), ptr_bert.get(k)) for k in ptr_bert.keys()])
            torch.save(ptr_bert, ptr_bert_path)
            print("Processing is done")
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
            print("Processing is done")
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
        ptr_config.save("conf/pretrained/skt.json")

    if args.type == 'etri':
        if not ptr_dir.exists():
            ptr_dir.mkdir(parents=True)

        zipfile_path = ptr_dir / "etri.zip"

        if not zipfile_path.exists():
            url = "https://drive.google.com/uc?id=1qVY-zZc2O2OliGNUwWClhcqJkLG_6uoD"
            gdown.download(url, output=str(zipfile_path))

            with zipfile.ZipFile(str(zipfile_path)) as unzip:
                unzip.extractall(str(ptr_dir))

            from pretrained.etri.tokenization import BertTokenizer as ETRITokenizer
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
            ptr_config.save("conf/pretrained/etri.json")
            print("Processing is done")
        else:
            print('Already you have relevant files of etri BERT')




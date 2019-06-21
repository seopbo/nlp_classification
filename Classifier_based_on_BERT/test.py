import torch
import pandas as pd
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertConfig, BertForSequenceClassification
from src_tokenizer.tokenization import BertTokenizer

# load tokenizer (good)
tokenizer = BertTokenizer.from_pretrained('./vocab.korean.rawtext.list', do_lower_case=False)

# restore pre-trained weight
ckpt = torch.load('./pytorch_model.bin') # pretrained weight from BertForPreTraining

config = BertConfig('./bert_config.json')
model = BertForSequenceClassification(config, num_labels=2)
model.load_state_dict(ckpt, strict=False)

# input preprocessing
tr_dataset = pd.read_csv('./data/train.txt', sep='\t')
example = tr_dataset.iloc[0]['document']

z=tokenizer.convert_tokens_to_ids(tokenizer.tokenize(example))
z=torch.tensor(z).reshape(1,-1)
model(z)
dir(BertPreTrainedModel)
import torch.nn as nn
from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel


class SentenceClassifier(BertPreTrainedModel):
    def __init__(self, config, num_classes, vocab) -> None:
        super(SentenceClassifier, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_classes)
        self.vocab = vocab
        self.init_weights()

    def forward(self, input_ids):
        # pooled_output is not same hidden vector corresponds to first token from last encoded layers
        attention_mask = input_ids.ne(self.vocab.to_indices(self.vocab.padding_token)).float()
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

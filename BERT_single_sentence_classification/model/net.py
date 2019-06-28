import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel


class BertClassifier(BertPreTrainedModel):
    def __init__(self, config, num_labels, vocab) -> None:
        super(BertClassifier, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.vocab = vocab
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, output_all_encoded_layers=False):
        # pooled_output is not same hidden vector corresponds to first token from last encoded layers
        attention_mask = input_ids.ne(self.vocab.to_indices(self.vocab.padding_token)).float()
        encoded_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                  output_all_encoded_layers=output_all_encoded_layers)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if output_all_encoded_layers:
            return logits, encoded_layers
        else:
            return logits

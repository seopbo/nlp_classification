import torch
import torch.nn as nn
import torch.nn.functional as F
from gluonnlp import Vocab

class SenCNN(nn.Module):
    """SenCNN class"""
    def __init__(self, num_classes: int, vocab: Vocab) -> None:
        """Instantiating SenCNN class

        Args:
            num_classes (int): the number of classes
            vocab (gluonnlp.Vocab): the instance of gluonnlp.Vocab
        """
        super(SenCNN, self).__init__()
        # static embedding
        self._static = nn.Embedding.from_pretrained(torch.from_numpy(vocab.embedding.idx_to_vec.asnumpy()),
                                                    freeze=True)
        # self.static.weight.data.copy_(torch.from_numpy(vocab.embedding.idx_to_vec.asnumpy()))
        # self.static.weight.requires_grad_(False)

        # non-static embedding
        self._non_static = nn.Embedding.from_pretrained(torch.from_numpy(vocab.embedding.idx_to_vec.asnumpy()),
                                                        freeze=False)
        # self.non_static = nn.Embedding(len(vocab), embedding_dim=300, padding_idx=0)
        # self.non_static.weight.data.copy_(torch.from_numpy(vocab.embedding.idx_to_vec.asnumpy()))

        # convolution layer
        self._tri_gram = nn.Conv1d(in_channels=300, out_channels=100, kernel_size=3)
        self._tetra_gram = nn.Conv1d(in_channels=300, out_channels=100, kernel_size=4)
        self._penta_gram = nn.Conv1d(in_channels=300, out_channels=100, kernel_size=5)

        # output layer
        self._fc = nn.Linear(in_features=300, out_features=num_classes)

        # dropout
        self._drop = nn.Dropout()

        # initialization
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # embedding layer
        static_batch = self._static(x)
        static_batch = static_batch.permute(0, 2, 1) # for Conv1d

        non_static_batch = self._non_static(x)
        non_static_batch = non_static_batch.permute(0, 2, 1) # for Conv1d

        # convolution layer (extract feature)
        tri_feature = F.relu(self._tri_gram(static_batch)) + F.relu(self._tri_gram(non_static_batch))
        tetra_feature = F.relu(self._tetra_gram(static_batch)) + F.relu(self._tetra_gram(non_static_batch))
        penta_feature = F.relu(self._penta_gram(static_batch)) + F.relu(self._penta_gram(non_static_batch))

        # max-overtime pooling
        tri_feature = torch.max(tri_feature, 2)[0]
        tetra_feature = torch.max(tetra_feature, 2)[0]
        penta_feature = torch.max(penta_feature, 2)[0]
        feature = torch.cat((tri_feature, tetra_feature, penta_feature), 1)

        # dropout
        feature = self._drop(feature)

        # output layer
        score = self._fc(feature)

        return score

    def _init_weights(self, layer) -> None:
        if isinstance(layer, nn.Conv1d):
            nn.init.kaiming_uniform_(layer.weight)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
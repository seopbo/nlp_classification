import torch
import torch.nn as nn
from model.ops import Flatten, Permute
from model.utils import Vocab


class CharCNN(nn.Module):
    """CharCNN class"""
    def __init__(self, num_classes: int, embedding_dim: int, vocab: Vocab) -> None:
        """Instantiating CharCNN class

        Args:
            num_classes (int): the number of classes
            embedding_dim (int): the dimension of embedding vector for token
            vocab (model.utils.Vocab): the instance of model.utils.Vocab
        """
        super(CharCNN, self).__init__()
        self._extractor = nn.Sequential(nn.Embedding(len(vocab), embedding_dim, vocab.to_indices(vocab.padding_token)),
                                        Permute(),
                                        nn.Conv1d(in_channels=embedding_dim, out_channels=256, kernel_size=7),
                                        nn.ReLU(),
                                        nn.MaxPool1d(3, 3),
                                        nn.Conv1d(in_channels=256, out_channels=256, kernel_size=7),
                                        nn.ReLU(),
                                        nn.MaxPool1d(3, 3),
                                        nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3),
                                        nn.ReLU(),
                                        nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3),
                                        nn.ReLU(),
                                        nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3),
                                        nn.ReLU(),
                                        nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3),
                                        nn.ReLU(),
                                        nn.MaxPool1d(3, 3),
                                        Flatten())

        self._classifier = nn.Sequential(nn.Linear(in_features=1792, out_features=512),
                                         nn.ReLU(),
                                         nn.Dropout(),
                                         nn.Linear(in_features=512, out_features=512),
                                         nn.ReLU(),
                                         nn.Dropout(),
                                         nn.Linear(in_features=512, out_features=num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = self._extractor(x)
        score = self._classifier(feature)
        return score

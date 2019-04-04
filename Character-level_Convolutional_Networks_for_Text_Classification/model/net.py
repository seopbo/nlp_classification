import torch
import torch.nn as nn


class Flatten(nn.Module):
    """Flatten class"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class Permute(nn.Module):
    """Permute class"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 1)


class CharCNN(nn.Module):
    """CharCNN class"""
    def __init__(self, num_classes: int, embedding_dim: int, dic: dict) -> None:
        """Instantiating CharCNN

        Args:
            num_classes (int): number of classes
            embedding_dim (int): embedding dimension of token
            dic (dict): token2idx
        """
        super(CharCNN, self).__init__()
        self._extractor = nn.Sequential(nn.Embedding(len(dic), embedding_dim, padding_idx=0),
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

        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor, p=.5) -> torch.Tensor:
        feature = self._extractor(x)
        score = self._classifier(feature)
        return score

    def _init_weights(self, layer) -> None:
        if isinstance(layer, nn.Conv1d):
            nn.init.kaiming_uniform_(layer.weight)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
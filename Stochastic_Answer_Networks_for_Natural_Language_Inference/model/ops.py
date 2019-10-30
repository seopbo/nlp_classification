import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence
from model.utils import Vocab
from typing import Tuple, Union, Callable


class Embedding(nn.Module):
    """Embedding class"""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int = 1,
        permuting: bool = True,
        tracking: bool = True,
    ) -> None:
        """Instantiating Embedding class

        Args:
            num_embeddings (int): the number of vocabulary size
            embedding_dim (int): the dimension of embedding vector
            padding_idx (int): denote padding_idx to "<pad>" token
            permuting (bool): permuting (n, l, c) -> (n, c, l). Default: True
            tracking (bool): tracking length of sequence. Default: True
        """
        super(Embedding, self).__init__()
        self._tracking = tracking
        self._permuting = permuting
        self._padding_idx = padding_idx
        self._ops = nn.Embedding(num_embeddings, embedding_dim, self._padding_idx)

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        fmap = self._ops(x).permute(0, 2, 1) if self._permuting else self._ops(x)

        if self._tracking:
            fmap_length = x.ne(self._padding_idx).sum(dim=1)
            return fmap, fmap_length
        else:
            return fmap


class PreEmbedding(nn.Module):
    """PreEmbedding class"""

    def __init__(
        self,
        vocab: Vocab,
        padding_idx: int = 1,
        freeze: bool = True,
        permuting: bool = True,
        tracking: bool = True,
    ) -> None:
        """Instantiating PreEmbedding class

        Args:
            vocab (model.utils.Vocab): the instance of model.utils.Vocab
            padding_idx (int): denote padding_idx to padding token
            freeze (bool): freezing weights. Default: False
            permuting (bool): permuting (n, l, c) -> (n, c, l). Default: True
            tracking (bool): tracking length of sequence. Default: True
        """
        super(PreEmbedding, self).__init__()
        self._padding_idx = padding_idx
        self._permuting = permuting
        self._tracking = tracking
        self._ops = nn.Embedding.from_pretrained(
            torch.from_numpy(vocab.embedding),
            freeze=freeze,
            padding_idx=self._padding_idx,
        )

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        fmap = self._ops(x).permute(0, 2, 1) if self._permuting else self._ops(x)

        if self._tracking:
            fmap_length = x.ne(self._padding_idx).sum(dim=1)
            return fmap, fmap_length
        else:
            return fmap


class Conv1d(nn.Module):
    """Conv1d class"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 1,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.relu,
        tracking: bool = True,
    ) -> None:
        """Instantiating Conv1d class
        Args:
            in_channels (int): the number of channels in the input feature map
            out_channels (int): the number of channels in the output feature emap
            kernel_size (int): the size of the convolving kernel
            stride (int): stride of the convolution. Default: 1
            padding (int): zero-padding added to both sides of the input. Default: 1
            activation (function): activation function. Default: F.relu
            tracking (bool): tracking length of sequence. Default: True
        """
        super(Conv1d, self).__init__()
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._ops = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self._activation = activation
        self._tracking = tracking

    def forward(
        self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self._tracking:
            fmap, fmap_length = x
            fmap_length = (
                fmap_length + 2 * self._padding - (self._kernel_size - 1) - 1
            ) / self._stride + 1
            fmap = (
                self._activation(self._ops(fmap))
                if self._activation is not None
                else self._ops(fmap)
            )
            return fmap, fmap_length
        else:
            fmap = (
                self._activation(self._ops(x))
                if self._activation is not None
                else self._ops(x)
            )
            return fmap


class Linker(nn.Module):
    """Linker class"""

    def __init__(self, permuting: bool = True):
        """Instantiating Linker class
        Args:
            permuting (bool): permuting (n, c, l) -> (n, l, c). Default: True
        """
        super(Linker, self).__init__()
        self._permuting = permuting

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> PackedSequence:
        fmap, fmap_length = x
        fmap = fmap.permute(0, 2, 1) if self._permuting else fmap
        return pack_padded_sequence(
            fmap, fmap_length, batch_first=True, enforce_sorted=False
        )


class BiLSTM(nn.Module):
    """BiLSTM class"""
    def __init__(self, input_size: int, hidden_size: int, using_sequence: bool = True) -> None:
        """Instantiating BiLSTM class
        Args:
            input_size (int): the number of expected features in the input x
            hidden_size (int): the number of features in the hidden state h
            using_sequence (bool): using all hidden states of sequence. Default: True
        """
        super(BiLSTM, self).__init__()
        self._using_sequence = using_sequence
        self._ops = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, x: PackedSequence) -> torch.Tensor:
        outputs, hc = self._ops(x)

        if self._using_sequence:
            hiddens = pad_packed_sequence(outputs)[0].permute(1, 0, 2)
            return hiddens
        else:
            feature = torch.cat([*hc[0]], dim=1)
            return feature


class MaxOut(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MaxOut, self).__init__()
        self._ops_1 = nn.Linear(input_size, hidden_size)
        self._ops_2 = nn.Linear(input_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_1 = self._ops_1(x)
        feature_2 = self._ops_2(x)
        return feature_1.max(feature_2)


class LexiconEncoder(nn.Module):
    def __init__(self, coarse_vocab, fine_vocab, fine_embedding_dim):
        super(LexiconEncoder, self).__init__()
        self._coarse_emb = PreEmbedding(coarse_vocab, coarse_vocab.to_indices(coarse_vocab.padding_token),
                                        freeze=False, permuting=False, tracking=True)
        self._fine_emb = Embedding(len(fine_vocab), fine_embedding_dim, fine_vocab.to_indices(fine_vocab.padding_token),
                                   permuting=True, tracking=False)
        self._conv_uni = Conv1d(in_channels=fine_embedding_dim, out_channels=50, kernel_size=1, stride=1,
                                padding=0, tracking=False)
        self._conv_tri = Conv1d(in_channels=fine_embedding_dim, out_channels=100, kernel_size=3, stride=1,
                                padding=0, tracking=False)
        self._conv_penta = Conv1d(in_channels=fine_embedding_dim, out_channels=150, kernel_size=5, stride=1,
                                  padding=0, tracking=False)
        self._output_size = self._coarse_emb._ops.embedding_dim + 50 + 100 + 150
        self._postion_wise_ffn_1 = Conv1d(in_channels=self._output_size,
                                          out_channels=self._output_size,
                                          kernel_size=1, stride=1, padding=0, tracking=False)
        self._postion_wise_ffn_2 = Conv1d(in_channels=self._output_size,
                                          out_channels=self._output_size,
                                          kernel_size=1, stride=1, padding=0, tracking=False)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        coarse_input, fine_input = inputs
        coarse_embed, length = self._coarse_emb(coarse_input)

        fine_input_reshaped = torch.cat([*fine_input], dim=0)
        fine_embed = self._fine_emb(fine_input_reshaped)

        fine_uni_fmap = self._conv_uni(fine_embed).max(dim=-1)[0]
        fine_uni_fmap = torch.stack(fine_uni_fmap.chunk(fine_input.size(0), dim=0))
        fine_tri_fmap = self._conv_tri(fine_embed).max(dim=-1)[0]
        fine_tri_fmap = torch.stack(fine_tri_fmap.chunk(fine_input.size(0), dim=0))
        fine_penta_fmap = self._conv_penta(fine_embed).max(dim=-1)[0]
        fine_penta_fmap = torch.stack(fine_penta_fmap.chunk(fine_input.size(0), dim=0))
        fine_fmap = torch.cat([fine_uni_fmap, fine_tri_fmap, fine_penta_fmap], dim=-1)
        fmap = torch.cat([coarse_embed, fine_fmap], dim=-1).permute(0, 2, 1)
        intermediate_fmap = self._postion_wise_ffn_1(fmap)
        lexicon_fmap = self._postion_wise_ffn_2(intermediate_fmap)
        return lexicon_fmap, length


class ContextualEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ContextualEncoder, self).__init__()
        self._link = Linker()
        self._ops_1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=True,
                              bidirectional=True)
        self._act_1 = MaxOut(input_size=hidden_size * 2, hidden_size=hidden_size)
        self._ops_2 = nn.LSTM(input_size=hidden_size * 2, hidden_size=hidden_size, num_layers=2, batch_first=True,
                              bidirectional=True)
        self._act_2 = MaxOut(input_size=hidden_size *2, hidden_size=hidden_size)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]):
        sequences = self._link(inputs)
        outputs_1, _ = self._ops_1(sequences)
        hidden_states_1, _ = pad_packed_sequence(outputs_1, batch_first=True)
        hidden_states_1 = self._act_1(hidden_states_1)
        outputs_2, _ = self._ops_2(outputs_1)
        hidden_states_2, length = pad_packed_sequence(outputs_2, batch_first=True)
        hidden_states_2 = self._act_2(hidden_states_2)
        contextual_fmap = torch.cat([hidden_states_1, hidden_states_2], dim=-1)
        return contextual_fmap, length

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from model.ops import MultiChannelEmbedding, ConvolutionLayer, MaxOverTimePooling
from gluonnlp import Vocab


class SenCNN(keras.Model):
    """SenCNN class"""

    def __init__(self, num_classes: int, vocab: Vocab) -> None:
        """Instantiating SenCNN class

        Args:
            num_classes (int): the number of classes
            vocab (gluonnlp.Vocab): the instance of gluonnlp.Vocab
        """
        super(SenCNN, self).__init__()
        self._embedding = MultiChannelEmbedding(vocab)
        self._convolution = ConvolutionLayer(300)
        self._pooling = MaxOverTimePooling()
        self._dropout = layers.Dropout(.5)
        self._fc = layers.Dense(units=num_classes)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        fmap = self._embedding(x)
        fmap = self._convolution(fmap)
        feature = self._pooling(fmap)
        feature = self._dropout(feature)
        score = self._fc(feature)
        return score



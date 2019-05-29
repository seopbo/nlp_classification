import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Model
from gluonnlp import Vocab
from typing import Tuple


class MultiChannelEmbedding(Model):
    """MultiChannelEmbedding class"""
    def __init__(self, vocab: Vocab) -> None:
        """Instantiating MultiChannelEmbedding class

        Args:
            vocab (gluonnlp.Vocab): the instance of gluonnlp.Vocab class
        """
        super(MultiChannelEmbedding, self).__init__()
        self._non_static_embedding = tf.Variable(initial_value=vocab.embedding.idx_to_vec.asnumpy(), trainable=True)
        self._static_embedding = tf.Variable(initial_value=vocab.embedding.idx_to_vec.asnumpy(), trainable=False)

    def call(self, idx: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        non_static_embedding = tf.nn.embedding_lookup(self._non_static_embedding, idx)
        static_embedding = tf.nn.embedding_lookup(self._static_embedding, idx)
        return non_static_embedding, static_embedding


class ConvolutionLayer(Model):
    """ConvolutionLayer class"""
    def __init__(self, filters: int = 300) -> None:
        """Instantiating ConvolutionLayer class

        Args:
            filters (int): the number of out channels
        """
        super(ConvolutionLayer, self).__init__()
        self._tri_gram_ops = keras.layers.Conv1D(filters=filters // 3, kernel_size=3, activation=tf.nn.relu)
        self._tetra_gram_ops = keras.layers.Conv1D(filters=filters // 3, kernel_size=4, activation=tf.nn.relu)
        self._penta_gram_ops = keras.layers.Conv1D(filters=filters // 3, kernel_size=5, activation=tf.nn.relu)

    def call(self, x: Tuple[tf.Tensor, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        non_static_embedding, static_embedding = x
        tri_fmap = self._tri_gram_ops(non_static_embedding) + self._tri_gram_ops(static_embedding)
        tetra_fmap = self._tetra_gram_ops(non_static_embedding) + self._tetra_gram_ops(static_embedding)
        penta_fmap = self._penta_gram_ops(non_static_embedding) + self._penta_gram_ops(static_embedding)
        return tri_fmap, tetra_fmap, penta_fmap


class MaxOverTimePooling(Model):
    """MaxOverTimePooling"""
    def call(self, x: Tuple[tf.Tensor, tf.Tensor, tf.Tensor]) -> tf.Tensor:
        tri_fmap, tetra_fmap, penta_fmap = x
        fmap = tf.concat([tf.reduce_max(tri_fmap, 1), tf.reduce_max(tetra_fmap, 1), tf.reduce_max(penta_fmap, 1)],
                         axis=-1)
        return fmap

import tensorflow as tf
from tensorflow.keras.layers import Layer, Concatenate


class WeightedAverage(Layer):

    def __init__(self):
        super(WeightedAverage, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(1, 1, len(input_shape)),
            initializer='uniform',
            dtype=tf.float32,
            trainable=True)

    def call(self, inputs):
        # inputs is a list of tensor of shape [(n_batch, n_feat), ..., (n_batch, n_feat)]
        # expand last dim of each input passed [(n_batch, n_feat, 1), ..., (n_batch, n_feat, 1)]
        inputs = [tf.expand_dims(i, -1) for i in inputs]
        inputs = Concatenate(axis=-1)(inputs)  # (n_batch, n_feat, n_inputs)
        weights = tf.nn.softmax(self.W, axis=-1)  # (1,1,n_inputs)
        # weights sum up to one on last dim

        return tf.reduce_sum(weights * inputs, axis=-1)  # (n_batch, n_feat)

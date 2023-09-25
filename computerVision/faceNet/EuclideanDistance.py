import tensorflow as tf
from tensorflow.keras.layers import Layer


class EuclideanDistance(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        # Unpack the vectors into separate lists
        featsA, featsB = inputs
        # Compute the sum of squared distances between the vectors
        sumSquared = tf.keras.backend.sum(tf.keras.backend.square(featsA - featsB), axis=1, keepdims=True)
        # Return the euclidean distance between the vectors
        return tf.keras.backend.sqrt(tf.keras.backend.maximum(sumSquared, tf.keras.backend.epsilon()))

    def get_config(self):
        config = super().get_config()
        return config

# File        :   siameseBranch.py
# Version     :   0.9.0
# Description :   Implements one siamese branch of the faceNet architecture

# Date:       :   Jun 04, 2023
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization


def euclideanDistance(vectors):
    # Unpack the vectors into separate lists
    (featsA, featsB) = vectors

    # Compute the sum of squared distances between the vectors
    sumSquared = tf.keras.backend.sum(tf.keras.backend.square(featsA - featsB), axis=1,
                                      keepdims=True)
    # Return the euclidean distance between the vectors
    return tf.keras.backend.sqrt(tf.keras.backend.maximum(sumSquared, tf.keras.backend.epsilon()))


def buildSiameseBranch(inputShape, embeddingDim=100):

    # Set the input axis order (channel order):
    chanDim = -1

    # [0] Input to the network:
    imageInput = Input(shape=inputShape)

    # == [1] Convolutional layers:
    conv1 = Conv2D(filters=32, kernel_size=5, strides=2, padding="same", activation="relu",
                   kernel_initializer="he_normal")(imageInput)
    conv1 = BatchNormalization(axis=chanDim)(conv1)

    # [1] Max Pooling:
    max1 = MaxPooling2D()(conv1)
    # [1] Drop out:
    drop1 = Dropout(rate=0.3)(max1)

    # == [2] Convolutional Layers:
    conv2 = Conv2D(filters=64, kernel_size=3, padding="same", activation="relu",
                   kernel_initializer="he_normal")(drop1)
    conv2 = BatchNormalization(axis=chanDim)(conv2)

    # [2] Max Pooling:
    max2 = MaxPooling2D()(conv2)
    # [2] Drop out:
    drop3 = Dropout(rate=0.3)(max2)

    # == [3] Convolutional Layers:
    conv3 = Conv2D(filters=128, kernel_size=3, padding="same", activation="relu",
                   kernel_initializer="he_normal")(drop3)
    conv3 = BatchNormalization(axis=chanDim)(conv3)

    # [2] Max Pooling:
    max3 = MaxPooling2D()(conv3)
    # [2] Drop out:
    drop4 = Dropout(rate=0.3)(max3)

    # == [4] Convolutional layers:
    conv4 = Conv2D(filters=256, kernel_size=3, padding="same", activation="relu",
                   kernel_initializer="he_normal")(drop4)
    conv4 = BatchNormalization(axis=chanDim)(conv4)

    # [4] Max Pooling:
    max4 = MaxPooling2D()(conv4)
    # [4] Drop out:
    drop5 = Dropout(rate=0.3)(max4)

    # Pooling/Flatten:
    globalPooled = GlobalAveragePooling2D()(drop5)

    # == Fully connected layers:
    # [1] Dense:
    dense1 = Dense(1024, activation="relu", kernel_initializer="he_normal")(globalPooled)
    dense1 = BatchNormalization()(dense1)
    drop6 = Dropout(rate=0.3)(dense1)

    # [2] Dense:
    dense2 = Dense(512, activation="relu", kernel_initializer="he_normal")(drop6)
    dense2 = BatchNormalization()(dense2)
    drop6 = Dropout(rate=0.3)(dense2)

    # Dense (Image Embeddings). Expected shape (None, embeddingDim)
    embeddingsOutput = Dense(embeddingDim)(drop6)

    # Set the model inputs/outputs:
    model = Model(inputs=imageInput, outputs=embeddingsOutput)

    # Done:
    return model
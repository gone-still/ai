# File        :   faceNet.py
# Version     :   0.7.5
# Description :   faceNet CNN architecture

# Date:       :   Jul 03, 2023
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dot

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
# from tensorflow.keras.layers import Activation
# from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.optimizers import Nadam

from tensorflow.keras.layers import Lambda

from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

from faceNetFunctions import euclideanDistance


class faceNet:

    @staticmethod
    def build(height, width, depth, namesList, embeddingDim=50, alpha=0.001, epochs=10):
        # Set the input axis order (channel order):
        inputShape = (height, width, depth)
        chanDim = -1

        # [0] Inputs to the network:
        image1 = Input(name="image1", shape=inputShape)
        image2 = Input(name="image2", shape=inputShape)

        # [1] Convolutional layers:
        conv1 = Conv2D(filters=32, kernel_size=5, strides=2, padding="same", activation="relu", kernel_initializer="he_normal")(image1)
        conv2 = Conv2D(filters=32, kernel_size=5, strides=2, padding="same", activation="relu", kernel_initializer="he_normal")(image2)

        # [1] Max Pooling:
        max1 = MaxPooling2D()(conv1)
        max2 = MaxPooling2D()(conv2)

        # [1] Drop out:
        drop1 = Dropout(rate=0.1)(max1)
        drop2 = Dropout(rate=0.1)(max2)

        # [2] Convolutional Layers:
        conv3 = Conv2D(filters=64, kernel_size=3, padding="same", activation="relu", kernel_initializer="he_normal")(drop1)
        conv4 = Conv2D(filters=64, kernel_size=3, padding="same", activation="relu", kernel_initializer="he_normal")(drop2)

        # [2] Max Pooling:
        max3 = MaxPooling2D()(conv3)
        max4 = MaxPooling2D()(conv4)

        # [2] Drop out:
        drop3 = Dropout(rate=0.1)(max3)
        drop4 = Dropout(rate=0.1)(max4)

        # [3] Convolutional Layers:
        conv5 = Conv2D(filters=128, kernel_size=3, padding="same", activation="relu", kernel_initializer="he_normal")(drop3)
        conv6 = Conv2D(filters=128, kernel_size=3, padding="same", activation="relu", kernel_initializer="he_normal")(drop4)

        # [2] Max Pooling:
        max5 = MaxPooling2D()(conv5)
        max6 = MaxPooling2D()(conv6)

        drop5 = Dropout(rate=0.1)(max5)
        drop6 = Dropout(rate=0.1)(max6)

        conv7 = Conv2D(filters=256, kernel_size=3, padding="same", activation="relu", kernel_initializer="he_normal")(drop5)
        conv8 = Conv2D(filters=256, kernel_size=3, padding="same", activation="relu", kernel_initializer="he_normal")(drop6)

        # [2] Max Pooling:
        max7 = MaxPooling2D()(conv7)
        max8 = MaxPooling2D()(conv8)

        # [2] Drop out:
        drop7 = Dropout(rate=0.1)(max7)
        drop8 = Dropout(rate=0.1)(max8)

        # Pooling/Flatten:
        globalPooled1 = GlobalAveragePooling2D()(drop7)
        globalPooled2 = GlobalAveragePooling2D()(drop8)

        # flattened1 = Flatten()(drop5)
        # flattened2 = Flatten()(drop6)

        # [3] Drop out:
        # drop7 = Dropout(rate=0.3)(globalPooled1)
        # drop8 = Dropout(rate=0.3)(globalPooled2)

        # Dense (Image Embeddings). Expected shape (None, embeddingDim)
        dense1 = Dense(1024, activation="relu", kernel_initializer="he_normal")(globalPooled1)
        dense2 = Dense(1024, activation="relu", kernel_initializer="he_normal")(globalPooled2)

        dense1 = BatchNormalization()(dense1)
        dense2 = BatchNormalization()(dense2)

        drop9 = Dropout(rate=0.1)(dense1)
        drop10 = Dropout(rate=0.1)(dense2)

        dense3 = Dense(512, activation="relu", kernel_initializer="he_normal")(drop9)
        dense4 = Dense(512, activation="relu", kernel_initializer="he_normal")(drop10)

        dense3 = BatchNormalization()(dense3)
        dense4 = BatchNormalization()(dense4)

        drop11 = Dropout(rate=0.1)(dense3)
        drop12 = Dropout(rate=0.1)(dense4)

        dense5 = Dense(embeddingDim)(drop11)
        dense6 = Dense(embeddingDim)(drop12)

        # def euclideanDistance(vectors):
        #     # unpack the vectors into separate lists
        #     (featsA, featsB) = vectors
        #
        #     # compute the sum of squared distances between the vectors
        #     sumSquared = tf.keras.backend.sum(tf.keras.backend.square(featsA - featsB), axis=1,
        #                                       keepdims=True)
        #     # return the euclidean distance between the vectors
        #     return tf.keras.backend.sqrt(tf.keras.backend.maximum(sumSquared, tf.keras.backend.epsilon()))

        distance = Lambda(euclideanDistance)([dense5, dense6])

        # Cosine Distance. Expected shape (None, 1)
        # distance = Dot(name="dot_product", normalize=True, axes=1)([dense1, dense2])
        # distance = Lambda(cosine_distance)([dense1, dense2])

        # Reshape to be a single number (shape will be (None, 1))
        # reshaped = Reshape(target_shape=[1])(dotProduct)

        # Dense. Expected shape: (None, 1)
        finalOutput = Dense(1, activation="sigmoid")(distance)

        # Set the model inputs/outputs:
        model = Model(inputs=[image1, image2], outputs=finalOutput)
        # Set the optimizer:
        optimizer = Adamax(learning_rate=alpha)

        # Compile the model:
        model.compile(loss="binary_crossentropy", optimizer=optimizer,
                      metrics=["accuracy"])

        # Done:
        return model

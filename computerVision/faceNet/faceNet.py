# File        :   faceNet.py
# Version     :   0.7.1
# Description :   faceNet CNN architecture

# Date:       :   Jul 01, 2023
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT

# File        :   faceNet.py
# Version     :   0.5.0
# Description :   faceNet CNN architecture

# Date:       :   Jun 13, 2023
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dot

# from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.layers import Lambda

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
        conv1 = Conv2D(filters=64, kernel_size=5, strides=2, padding="same", activation="relu")(image1)
        conv2 = Conv2D(filters=64, kernel_size=5, strides=2, padding="same", activation="relu")(image2)

        # [1] Max Pooling:
        max1 = MaxPooling2D()(conv1)
        max2 = MaxPooling2D()(conv2)

        # [1] Drop out:
        drop1 = Dropout(rate=0.1)(max1)
        drop2 = Dropout(rate=0.1)(max2)

        # [2] Convolutional Layers:
        conv3 = Conv2D(filters=128, kernel_size=3, padding="same", activation="relu")(drop1)
        conv4 = Conv2D(filters=128, kernel_size=3, padding="same", activation="relu")(drop2)

        # [2] Max Pooling:
        max3 = MaxPooling2D()(conv3)
        max4 = MaxPooling2D()(conv4)

        # [2] Drop out:
        drop3 = Dropout(rate=0.1)(max3)
        drop4 = Dropout(rate=0.1)(max4)

        # [3] Convolutional Layers:
        conv5 = Conv2D(filters=256, kernel_size=3, padding="same", activation="relu")(drop3)
        conv6 = Conv2D(filters=256, kernel_size=3, padding="same", activation="relu")(drop4)

        # [2] Max Pooling:
        max5 = MaxPooling2D()(conv5)
        max6 = MaxPooling2D()(conv6)

        # [2] Drop out:
        drop5 = Dropout(rate=0.1)(max5)
        drop6 = Dropout(rate=0.1)(max6)

        # Pooling/Flatten:
        globalPooled1 = GlobalAveragePooling2D()(drop5)
        globalPooled2 = GlobalAveragePooling2D()(drop6)

        # [1] First Dense with 256 units. Expected shape (None, 256)
        dense1 = Dense(256, activation="relu")(globalPooled1)
        dense2 = Dense(256, activation="relu")(globalPooled2)

        # [3] Droput:
        drop7 = Dropout(rate=0.1)(dense1)
        drop8 = Dropout(rate=0.1)(dense2)

        # [2] Second Dense (Image Embeddings). Expected shape (None, embeddingDim)
        dense3 = Dense(embeddingDim)(drop7)
        dense4 = Dense(embeddingDim)(drop8)

        distance = Lambda(euclideanDistance)([dense3, dense4])

        # Cosine Distance. Expected shape (None, 1)
        # distance = Dot(name="dot_product", normalize=True, axes=1)([dense1, dense2])
        # distance = Lambda(cosine_distance)([dense1, dense2])

        # [3] Output Dense. Expected shape: (None, 1)
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

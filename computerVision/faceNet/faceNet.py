# File        :   faceNet.py
# Version     :   0.5.0
# Description :   faceNet CNN architecture

# Date:       :   Jun 13, 2023
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dot

# from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
# from tensorflow.keras.layers import Activation
# from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape

from tensorflow.keras.optimizers import Adam


class faceNet:
    @staticmethod
    def build(height, width, depth, embeddingDim=50, alpha=0.001, epochs=10):
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
        drop1 = Dropout(rate=0.3)(max1)
        drop2 = Dropout(rate=0.3)(max2)

        # [2] Convolutional Layers:
        conv3 = Conv2D(filters=128, kernel_size=3, padding="same", activation="relu")(drop1)
        conv4 = Conv2D(filters=128, kernel_size=3, padding="same", activation="relu")(drop2)

        # [2] Max Pooling:
        max3 = MaxPooling2D()(conv3)
        max4 = MaxPooling2D()(conv4)

        # [2] Drop out:
        drop3 = Dropout(rate=0.3)(max3)
        drop4 = Dropout(rate=0.3)(max4)

        # [3] Convolutional Layers:
        conv5 = Conv2D(filters=256, kernel_size=3, padding="same", activation="relu")(drop3)
        conv6 = Conv2D(filters=256, kernel_size=3, padding="same", activation="relu")(drop4)

        # [2] Max Pooling:
        max5 = MaxPooling2D()(conv5)
        max6 = MaxPooling2D()(conv6)

        # [2] Drop out:
        drop5 = Dropout(rate=0.3)(max5)
        drop6 = Dropout(rate=0.3)(max6)

        # Pooling/Flatten:
        globalPooled1 = GlobalAveragePooling2D()(drop5)
        globalPooled2 = GlobalAveragePooling2D()(drop6)

        # flattened1 = Flatten()(drop5)
        # flattened2 = Flatten()(drop6)

        # [3] Drop out:
        # drop7 = Dropout(rate=0.3)(globalPooled1)
        # drop8 = Dropout(rate=0.3)(globalPooled2)

        # Dense (Image Embeddings). Expected shape (None, embeddingDim)
        dense1 = Dense(embeddingDim)(globalPooled1)
        dense2 = Dense(embeddingDim)(globalPooled2)

        # Cosine Distance. Expected shape (None, 1)
        dotProduct = Dot(name="dot_product", normalize=True, axes=1)([dense1, dense2])

        # Reshape to be a single number (shape will be (None, 1))
        reshaped = Reshape(target_shape=[1])(dotProduct)

        # Dense. Expected shape: (None, 1)
        finalOutput = Dense(1, activation="sigmoid")(reshaped)

        # Set the model inputs/outputs:
        model = Model(inputs=[image1, image2], outputs=finalOutput)
        # Set the optimizer:
        optimizer = Adam(lr=alpha, decay=alpha / (epochs * 0.5))

        # Compile the model:
        model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

        # Done:
        return model

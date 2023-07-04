# File        :   faceNet.py
# Version     :   0.8.5
# Description :   faceNet CNN architecture

# Date:       :   Jun 04, 2023
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dot

from tensorflow.keras.layers import Dense

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.optimizers import Nadam

from tensorflow.keras.layers import Lambda

from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

from siameseBranch import buildSiameseBranch, euclideanDistance


class faceNet:

    @staticmethod
    def build(height, width, depth, namesList, embeddingDim=100, alpha=0.001, epochs=10):
        # Set the input axis order (channel order):
        inputShape = (height, width, depth)
        chanDim = -1

        # Create input layers:
        image1 = Input(name=namesList[0], shape=inputShape)
        image2 = Input(name=namesList[1], shape=inputShape)

        # Get image embeddings by creating 2 siamese branches:
        getEmbeddings = buildSiameseBranch(inputShape=inputShape, embeddingDim=embeddingDim)

        # Get image embeddings:
        image1Embeddings = getEmbeddings(image1)
        image2Embeddings = getEmbeddings(image2)

        # Compute distance:
        distance = Lambda(euclideanDistance)([image1Embeddings, image2Embeddings])

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
        # optimizer = Adam(lr=alpha, decay=alpha / (epochs * 0.5))

        boundaries = [2023]
        values = [0.001, 0.001 * 0.8]

        lr_schedule = PiecewiseConstantDecay(boundaries, values)
        optimizer = Adamax(learning_rate=lr_schedule)

        def get_lr_metric(optimizer):
            def lr(y_true, y_pred):
                return optimizer._decayed_lr(tf.float32)  # I use ._decayed_lr method instead of .lr

            return lr

        lr_metric = get_lr_metric(optimizer)

        # Compile the model:
        model.compile(loss="binary_crossentropy", optimizer=optimizer,
                      metrics=["accuracy", lr_metric])

        # Done:
        return model

# File        :   faceNet.py
# Version     :   0.9.0
# Description :   faceNet CNN architecture

# Date:       :   Jun 05, 2023
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

from siameseBranch import buildSiameseBranch, euclideanDistance, getLearningRate
from WeightedAverage import WeightedAverage


class faceNet:

    @staticmethod
    # Builds the complete faceNet siamese network:
    def build(height, width, depth, namesList, embeddingDim=100, alpha=0.001, distanceCode="euclidean",
              lrSchedulerParameters=None):

        # Set the input axis order (channel order):
        inputShape = (height, width, depth)

        # Create input layers:
        image1 = Input(name=namesList[0], shape=inputShape)
        image2 = Input(name=namesList[1], shape=inputShape)

        # Get image embeddings by creating 2 siamese branches:
        getEmbeddings = buildSiameseBranch(inputShape=inputShape, embeddingDim=embeddingDim)

        # Get image embeddings:
        image1Embeddings = getEmbeddings(image1)
        image2Embeddings = getEmbeddings(image2)

        # Compute distance:
        if distanceCode == "euclidean":
            # Euclidean Distance. Expected shape (None, 1)
            distance = Lambda(euclideanDistance)([image1Embeddings, image2Embeddings])
            print("faceNet[build]>> Using Euclidean Distance")
        elif distanceCode == "cosine":
            # Cosine Distance. Expected shape (None, 1)
            distance = Dot(name="dot_product", normalize=True, axes=1)([image1Embeddings, image2Embeddings])
            print("faceNet[build]>> Using Cosine Distance")
        else:
            # Weighted average:
            distance1 = Lambda(euclideanDistance)([image1Embeddings, image2Embeddings])
            distance2 = Dot(name="dot_product", normalize=True, axes=1)([image1Embeddings, image2Embeddings])
            distance = WeightedAverage()([distance1, distance2])
            print("faceNet[build]>> Using Weighted Average of both distances")

        # Reshape to be a single number (shape will be (None, 1))
        # reshaped = Reshape(target_shape=[1])(dotProduct)

        # Dense. Expected shape: (None, 1)
        finalOutput = Dense(1, activation="sigmoid")(distance)

        # Set the model inputs/outputs:
        model = Model(inputs=[image1, image2], outputs=finalOutput)

        # Set the optimizer and learning rate scheduler:
        if lrSchedulerParameters:
            # Get the step boundaries:
            boundaries = lrSchedulerParameters[0]
            # Get the learning rate values:
            values = lrSchedulerParameters[1]

            # Set the scheduler:
            lrSchedule = PiecewiseConstantDecay(boundaries, values)
            print("faceNet>> Using learning rate scheduler, params:", boundaries, values)

            # Set the optimizer:
            optimizer = Adamax(learning_rate=lrSchedule)
            lrMetric = getLearningRate(optimizer)
            metrics = ["accuracy", lrMetric]  # , tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]

        else:
            optimizer = Adamax(learning_rate=alpha)
            metrics = ["accuracy"]  # , tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]

        # Compile the model:
        model.compile(loss="binary_crossentropy", optimizer=optimizer,
                      metrics=metrics)

        # Done:
        return model

# File        :   tfLinearClassifier.py
# Version     :   1.0.0
# Description :   Linear classifier implemented on tensorflow
# Date:       :   Sept 08, 2023
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# Define forward-pass function:
# Implements the linear transformation w x inputs + b
def model(inputs, W, b):
    return tf.matmul(inputs, W) + b


# Define the loss function - Mean Squared Error:
def squareLoss(targets, predictions):
    # This is a tensor with the same shape as targets
    # and predictions, containing per-sample loss scores:
    perSampleLosses = tf.square(targets - predictions)

    # Average these losses into a single scalar loss via
    # the reduce_mean function:
    return tf.reduce_mean(perSampleLosses)


# Define the training step.
# It receives training data and updates the weights W and b
# according to the minimum loss:
def trainingStep(inputs, targets, W, b, learningRate=0.1):
    # Forward pass inside the gradient tape to track the
    # loss and further compute its gradient:
    with tf.GradientTape() as tape:
        # Apply linear transform and get predictions:
        predictions = model(inputs, W, b)
        # Get error:
        loss = squareLoss(targets, predictions)
    # Retrive gradient of the loss with regard to weights:
    gradientW, gradientB = tape.gradient(loss, [W, b])
    # Update the weights:
    W.assign_sub(gradientW * learningRate)  # W_next = W_current - (gradient * alpha)
    b.assign_sub(gradientB * learningRate)  # b_next = b_current - (gradient * alpha)
    return loss


# Generate two classes of random points in a 2D plane:
samplesPerClass = 1000
# Negative class:
negativeSamples = np.random.multivariate_normal(
    mean=[0, 3],  # Mean located on (feature1, feature2)
    cov=[[1, 0.5], [0.5, 1]],  # Ovali-like distribution
    size=samplesPerClass
)
# Positive class:
positiveClass = np.random.multivariate_normal(
    mean=[3, 0],
    cov=[[1, 0.5], [0.5, 1]],
    size=samplesPerClass
)

# Vertically stack the two classes into one array,
# shape -> 2 (clases), 2000 (samples)
inputs = np.vstack((negativeSamples, positiveClass)).astype(np.float32)

# Generate labels:
targets = np.vstack((np.zeros((samplesPerClass, 1), dtype="float32"),
                     np.ones((samplesPerClass, 1), dtype="float32")))

# Plot the data:
plt.scatter(inputs[:, 0], inputs[:, 1], c=targets[:, 0])  # feature 1, feature 2, labels
plt.show()

# Create the linear classifier that implements the transformation prediction = w x input + b
# and minimices the square of the difference between the predictions and the target
inputDim = 2  # Features
outputDim = 1  # Label

# Create variable tensors with randomized values:
W = tf.Variable(initial_value=tf.random.uniform(shape=(inputDim, outputDim)))
b = tf.Variable(initial_value=tf.random.uniform(shape=(outputDim,)))

# Set the batch training loop, which performs batch training in all the dataset.
# Run it for 40 steps using a lr = 0.1

for step in range(40):
    loss = trainingStep(inputs, targets, W, b, 0.1)
    print(f"Loss at step {step}: {loss:.4f}")

# With the minimized weights, get the new predictions:
predictions = model(inputs, W, b)

# Plot classification results, class threshold -> 0.5 (0 if pred < 0.5, 1 otherwise)
# Plot decision boundary:
# Generate 100 regularly spaced numbers between -1 and 4 for line plotting:
x = np.linspace(-1, 4, 100)
# Line equation:
y = -(W[0] / W[1]) * x + (0.5 - b) / W[1]
plt.plot(x, y, "-r")
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
plt.show()

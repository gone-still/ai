import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Create 1000 samples with normal distribution:

totalSamples = 1000
features = 2

randomSeed = 42
np.random.seed(randomSeed)

centers = [(0, 3), (3, 0)]
stdDev = [1.0, 1.0]

# Get points and labels:
X, y = make_blobs(n_samples=totalSamples, centers=centers, n_features=features, cluster_std=stdDev,
                  random_state=randomSeed)

# Plot points:
plt.scatter(X[y == 0, 0], X[y == 0, 1], color="red", s=10, label="Cluster1")
plt.scatter(X[y == 1, 0], X[y == 1, 1], color="blue", s=10, label="Cluster2")
# plt.show()

# Expand y dimension:
y = np.expand_dims(y, axis=1)

# Model Equation:
# y = W * x + b

# Prepare W and b arrays:
# W has two parameters and is of shape (2,1) (One column):
W = np.random.rand(features, 1)
# b is a scalar parameter:
b = np.zeros((1,))

# Numpy arrays to tensor variable:
W = tf.Variable(W)
b = tf.Variable(b)

# Prepare training loop:
epochs = 100
learningRate = 0.1

for i in range(epochs):
    with tf.GradientTape() as tape:
        # Get new predictions:
        yNew = tf.matmul(X, W) + b

        # Compute the error (MSE):
        diff = tf.square(y - yNew)
        totalSum = tf.math.reduce_sum(diff, axis=0)
        loss = (1 / totalSamples) * totalSum

    # Get gradient of W and B:
    gradientW, gradientB = tape.gradient(loss, [W, b])

    # Update W and B:
    # parameterNew = parameterOld - (deltaParameter * lr)
    W.assign_add(-gradientW * learningRate)
    b.assign_add(-gradientB * learningRate)

    print("Epoch: ", i, "Loss:", np.array(loss))

# Classify new sample:
xTest = np.array([2.68, 3.48])
xTest = np.expand_dims(xTest, axis=0)

# Get new label:
yTest = tf.matmul(xTest, W) + b
yTest = np.array(yTest)

if yTest[0] < 0.5:
    yTest = 0
    yColor = "red"
else:
    yTest = 1
    yColor = "blue"
print("New Label", yTest, "Color", yColor)

# Line equation:
W = np.array(W)
b = np.array(b)
x = np.linspace(-1, 4, 10)
y = -(W[0][0] / W[1][0]) * x + (0.5 - b) / W[1][0]
plt.plot(x, y, "-r")
plt.scatter(xTest[0, 0], xTest[0, 1], color=yColor, s=10, edgecolors="black", label="New")
plt.show()

# File        :   kMeans.py
# Version     :   1.0.0
# Description :   Naive implementation of K-Means
#                
# Date:       :   Mar 26, 2023
# Author      :   Mr. X
# License     :   Creative Commons CC0

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets._samples_generator import make_blobs
import random

# Output path:
path = "D://k-Means//"

# Create random data samples:
# x -> the points, c - > cluster label
data, c = make_blobs(n_samples=70, centers=3, cluster_std=2.5, random_state=97792)

# Plot the data:
plt.scatter(data[:, 0], data[:, 1], s=20)
plt.show()

# Get total samples:
totalSamples = len(data)

# Create labels list:
labels = [-1] * totalSamples

# Random centroids location
# min and max range:
xMin = np.amin(data[:, 0])
xMax = np.amax(data[:, 0])

yMin = np.amin(data[:, 1])
yMax = np.amax(data[:, 1])

# Set initial clusters:
clusterColors = ["red", "blue", "black"]
k = 3
clusters = np.array([[0.0, 0.0]] * k)
for i in range(k):
    xRandom = random.uniform(xMin, xMax)
    yRandom = random.uniform(yMin, yMax)
    clusters[i] = [xRandom, yRandom]

# Plot clusters:
for c in range(k):
    # Get cluster
    currentPoint = clusters[c]

    # Plot cluster:
    clusterColor = clusterColors[c]
    plt.title("Initial Random Cluster Centroids")
    plt.scatter(data[:, 0], data[:, 1], s=20, c="green")
    plt.scatter(currentPoint[0], currentPoint[1], marker="x", s=20, c=clusterColor, label="Cluster: "+str(c))
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=k)

# plt.savefig(path + "kmeans_00.png", bbox_inches="tight", dpi=120)
plt.show()

# Iteration Counter:
iterationCounter = 0

# Min error threshold:
totalError = 0.0
epsilon = 0.001

# Stop flag:
divideSpace = True

while (divideSpace):

    # Temp clusters:
    tempClusters = np.array([[0.0, 0.0, 0.0]] * k)

    # Distance from each point to each cluster:
    for p in range(totalSamples):

        # Get point:
        currentPoint = data[p]
        x = currentPoint[0]
        y = currentPoint[1]

        minDistance = np.inf

        # Compute distance from current point
        # to all cluster means:
        for c in range(k):
            # Get cluster mean:
            currentCluster = clusters[c]
            cx = currentCluster[0]
            cy = currentCluster[1]

            # Compute (euclidean) distance:
            dx = pow(cx - x, 2.0)
            dy = pow(cy - y, 2.0)

            d = dx + dy

            # Only keep the cluster that produced
            # the minimum distance:
            if (d < minDistance):
                minDistance = d
                labels[p] = c

        # Get new cluster label, freshly assigned:
        newLabel = labels[p]

        # Store cluster x and y accumulation for
        # later median calculation:
        tempClusters[newLabel][0] = tempClusters[newLabel][0] + currentPoint[0]
        tempClusters[newLabel][1] = tempClusters[newLabel][1] + currentPoint[1]

        # Here, the numbers of samples so far
        # within the same cluster:
        tempClusters[newLabel][2] = tempClusters[newLabel][2] + 1

        # Plot the new data point, with corresponding cluster
        # label:
        clusterColor = clusterColors[newLabel]
        plt.scatter(currentPoint[0], currentPoint[1], s=20, c=clusterColor)

    # Compute mean for each cluster:
    for c in range(k):

        # Get x, y and counter for each cluster:
        currentEntry = tempClusters[c]
        x = currentEntry[0]
        y = currentEntry[1]

        # Division factor:
        factor = 1 / currentEntry[2]

        # Compute mean for this cluster:
        currentEntry[0] = factor * x
        currentEntry[1] = factor * y

    # Means are the new centroids,
    # Compute error:
    totalError = 0.0
    for c in range(k):

        # Get current means:
        oldMeans = clusters[c]
        oldX = oldMeans[0]
        oldY = oldMeans[1]

        # Get new means:
        newMeans = tempClusters[c]
        newX = newMeans[0]
        newY = newMeans[1]

        # Get distance from old to new:
        xDelta = pow(newX - oldX, 2.0)
        yDelta = pow(newY - oldY, 2.0)

        # Compute cluster to median error:
        error = xDelta + yDelta

        # Accumulate error:
        totalError = totalError + error
        print(error, totalError)

        # Set new Means as new clusters:
        clusters[c][0] = newX
        clusters[c][1] = newY

        # Plot cluster old and new
        # centroid:
        clusterColor = clusterColors[c]
        plt.title("Iteration: " + str(iterationCounter) + " Error: " + str(round(totalError, 4)))

        plt.scatter(oldX, oldY, marker="x", s=20, c=clusterColor, label = "Cluster: " + str(c))
        plt.scatter(newX, newY, marker="v",  edgecolor="c", s=40, c=clusterColor, label = "Centroid: " + str(c))

        plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2*k)

    # Increase iteration counter:
    iterationCounter += 1

    # Plot graph:
    # Save the plot before showing it:
    # plt.savefig(path + "kmeans_"+str(iterationCounter)+".png", bbox_inches="tight", dpi=120)
    plt.show()

    # Check if error is below threshold:
    if (totalError <= epsilon):
        divideSpace = False

# File        :   multiLinearRegression.py
# Version     :   1.0.0

# Description :   Multivariate Linear Regression from scratch using
#                 the house price predicted from house size and total 
#                 rooms dataset

# Date:       :   Mar 19, 2023
# Author      :   Ricardo Acevedo-Avila
# License     :   Creative Commons CC0

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Plot Data function:
def plotData(x, y, xLabel):
    plt.xlabel(xLabel)
    plt.ylabel("House Price")
    # plt.legend()
    plt.grid(True)
    plt.plot(x, y, "bo")
    plt.show()

# Plots the historical cost through all
# epochs:
def plotCostHistory(lossHistory):
    historyEpochs = []
    costPlot = []
    count = 0

    for i in lossHistory:
        costPlot.append(i[0][0])
        historyEpochs.append(count)
        count += 1

    costPlot = np.array(costPlot)
    historyEpochs = np.array(historyEpochs)

    plt.xlabel("Epochs")
    plt.ylabel("Cost")
    plt.plot(historyEpochs, costPlot, "m", linewidth="5")
    plt.show()

def plotFit(rawX, rawY, featureStats):
    # create a deep copy of the original Y data:
    linearModel = rawY.copy()

    # Get feature stats:
    featuresMean = featureStats[0]
    featuresStdDev = featureStats[1]

    # Get the fitted points from the original
    # testing labels:
    for s in range(houseSize.shape[0]):
        # Normalize features:
        x1 = (houseSize[s][0] - featuresMean[0]) / featuresStdDev[0]
        x2 = (roomSize[s][0] - featuresMean[0]) / featuresStdDev[0]
        # Predict the new cost:
        y = theta[0] + theta[1] * x1 + theta[2] * x2 \
            # Store the new Y predictions:
        linearModel[s] = y

    # Get the end and starting points of the
    # predicted values:
    yMin = np.amin(linearModel)
    yMax = np.amax(linearModel)

    xMin = np.amin(houseSize, axis=0)[0]
    xMax = np.amax(houseSize, axis=0)[0]

    # Check out the fitted line:
    plt.xlabel("House Size")
    plt.ylabel("House Price")
    plt.grid(True)
    plt.plot(houseSize, rawY, "bo")
    plt.plot(houseSize, linearModel, "go")

    # Set the starting and ending points of
    # the fitted line:
    y1 = [yMin, yMax]
    x1 = [xMin, xMax]
    plt.axline((xMin, yMin), (xMax, yMax), linewidth=1, color="r")
    plt.show()

# Dataset loading function:
def loadData(filename):
    # Feature names:
    columnNames = ["housesize", "rooms", "price"]
    # Dataset loading:
    dataFrame = pd.read_csv(filename, names=columnNames, sep=",", index_col=False)
    # Create the dataset as a numpy array:
    data = np.array(dataFrame, dtype=float)

    return data


# Feature normalization:
def featureNormalization(dataSet):
    # Local copy of the data set:
    localDataSet = dataSet.copy()
    # Get array shape:
    (rows, columns) = localDataSet.shape[:2]
    # Feature mean and std dev are stored here:
    featuresMean = []
    featuresStd = []
    # Normalize on a colum-by-colum
    # basis, exclude the prection column:
    for c in range(columns - 1):
        # Get current column:
        currentColumn = localDataSet[:, c]
        # Get feature mean and std dev:
        columnMean = np.mean(currentColumn, 0)
        columnStdDev = np.std(currentColumn, 0)
        # Into the arrays:
        featuresMean.append(columnMean)
        featuresStd.append(columnStdDev)
        # Normalize the entire column:
        localDataSet[:, c] = (localDataSet[:, c] - columnMean) / columnStdDev

    return (localDataSet, [featuresMean, featuresStd])


# Cost function
# Measures the error between the predicted values and
# the real values.
# implementation of loss function, mean squared error:
def costFunction(x, y, theta):
    totalSamples = y.shape[0]
    error = np.matmul(x, theta) - y
    factor = 1.0 / (2.0 * totalSamples)
    cost = factor * pow(error, 2.0)
    return cost


# Gradient Descent:
def gradientDescent(x, y, theta, alpha=0.1, epochs=10):
    # Get number of samples:
    totalSamples = x.shape[0]
    # Store all cost/loss/error through all the iterations:
    lossHistory = []

    for i in range(epochs):

        # Compute the error/loss:
        error = np.matmul(x, theta) - y
        factor = 1.0 / totalSamples

        # Compute the cost derivative/gradient:
        costDerivative =  np.matmul(x.transpose(), error)
        costDerivative = factor * costDerivative

        # Theta (optimal parameters) update is stored here, for
        # each epoch iteration:
        theta = theta - (alpha * costDerivative)
        # Cost/loss for all epoch iterations is stored here:
        lossHistory.append(costFunction(x, y, theta))

    return (theta, lossHistory)


# Hyperparameters:
alpha = 0.1
epochs = 50

# Dataset path:
path = "D://dataSets//house_price_data.txt"

# Load the dataset:
data = loadData(path)

# Extract features and prediction variable:
(x, y) = (data[:, :2], data[:, -1])
rawX = x.copy()
rawY = y.copy()

# Plot the data set
# X is a two-variable array: house size and room size,
# slice the two columns into individual arrays for plotting:
houseSize = x[:, 0:1]
roomSize = x[:, 1:2]

# Plot the two features vs prediction (house price):
plotData(houseSize, y, xLabel="House Size")
plotData(roomSize, y, xLabel="Room Size")

# Normalize the dataset:
(data, featureStats) = featureNormalization(data)

# Extract (normalized) features and prediction variable:
(x, y) = (data[:, :2], data[:, -1])

# Extract dataset shape: (rows - samples and cols - features):
(samples, features) = x.shape[:2]

# Does nothin to y:
y = np.reshape(y, (samples, 1))

# Adds an extra column at the first position of the array to
# allow matrix multiplication:
x = np.hstack((np.ones((samples, 1)), x))
# Initialize the theta matrix, this matrix

# Stores all the optimized values,
# should be an array of ( features + 1, 1)
# The multivariable lineal model is:
#  y = b + m1 * x1 + m2 * x2    or:
#  hθ(x) = theta[0] + theta[1] x1 + theta[2] x2
theta = np.zeros((features + 1, 1))

# Run gradient descent:
theta, lossHistory = gradientDescent(x, y, theta, alpha, epochs)
J = costFunction(x, y, theta)
# theta array is the line model -> hθ(x) = theta[0] + theta[1] x1 + theta[2] x2
print("Cost: ", J)
print("Parameters: ", theta)

# Plot the historical cost:
plotCostHistory(lossHistory)

# Check out the fitted line:
plotFit(rawX, rawY, featureStats)

# Testing a prediction:
testHouseSize = 1600
testRoomSize = 3

# Store the features in an array:
x = [testHouseSize, testRoomSize]

# Get feature stats:
featuresMean = featureStats[0]
featuresStdDev = featureStats[1]

# Normalize the test features:
x[0] = (x[0] - featuresMean[0]) / featuresStdDev[0]
x[1] = (x[1] - featuresMean[1]) / featuresStdDev[1]

# Predict the price via linear model of two features:
y = theta[0] + theta[1] * x[0] + theta[2] * x[1]

# Pretty-print the prediction
priceFormat = round(y[0], 4)
priceFormat = "{:,}".format(priceFormat)
print("Predicted price of house: ", priceFormat)

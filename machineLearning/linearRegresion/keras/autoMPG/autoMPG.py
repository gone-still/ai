# File        :   kerasLinearRegression-MPG.py
# Version     :   1.0.0
# Description :   Keras tutorial for a linear predictor using
#                 the MPG dataset
#
# Date:       :   Mar 17, 2023
# Author      :   François Chollet
# Modified by :   Ricardo Acevedo-Avila
# License     :   Creative Commons CC0

# Basic regression: Predict fuel efficiency from kera's
# tutorial web site: https://www.tensorflow.org/tutorials/keras/regression
# Adapted to run with tensorflow 2.40

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


# Training-Testing plot function:
def plot_loss(history):
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.ylim([0, 10])
    plt.xlabel("Epoch")
    plt.ylabel("Error [MPG]")
    plt.legend()
    plt.grid(True)
    plt.show()


# Plot of linear fit to the Horsepower-MPG example:
def plot_horsepower(x, y):
    plt.scatter(train_features["Horsepower"], train_labels, label="Data")
    plt.plot(x, y, color="k", label="Prediction")
    plt.xlabel("Horsepower")
    plt.ylabel("MPG")
    plt.legend()
    plt.show()


# Plot of linear regression vs true values:
def plot_linearRegression(test_labels, test_predictions):
    a = plt.axes(aspect="equal")
    plt.scatter(test_labels, test_predictions)
    # plt.plot(test_labels, test_predictions, color="k", label="Prediction")
    plt.xlabel("True Values [MPG]")
    plt.ylabel("Predictions [MPG]")
    lims = [0, 50]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.show()


# Print tensorflow version:
print(tf.__version__)

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

# Download the dataset:
# url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data" # Web
url = "D://dataSets//MPG//auto-mpg.data" # Local

# Feature names:
column_names = ["MPG", "Cylinders", "Displacement", "Horsepower", "Weight", "Acceleration", "Model Year", "Origin"]

# Dataset loading:
raw_dataset = pd.read_csv(url, names=column_names, na_values="?", comment="\t", sep=" ", skipinitialspace=True)
dataset = raw_dataset.copy()

# Print the last 5 samples/rows of the dataset, show all columns:
pd.set_option("display.max_columns", None)
print(dataset.tail())

# Check how many values in the dataset are unknown:
print(dataset.isna().sum())
print("------------------")
# Drop the samples/rows where features are unknow:
dataset = dataset.dropna()
print(dataset.isna().sum())

# One-hot encode the feature "Origin", as it is categorical,
# not numerical.
# "Origin" in the dataset is encoded as integer, each integer is a country
# Create a dictionary to map those integers to the actual country names:
dataset["Origin"] = dataset["Origin"].map({1: "USA", 2: "Europe", 3: "Japan"})

# One-hot encode the feature:
dataset = pd.get_dummies(dataset, columns=["Origin"], prefix="", prefix_sep="")
pd.set_option("display.max_columns", None)
print(dataset.tail())

# Split the dataset into a training set and a test set, using the
# 80% (train) and 20% (test) division rule:
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Separate the target featyre — the "label" — from the other features,
# the model will predict the "MPG" feature:
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop("MPG")
test_labels = test_features.pop("MPG")

# Normalize the data via a normalization layer, normalize across
# the columns:

# normalizer = tf.keras.layers.LayerNormalization(axis=-1)
normalizer = tf.keras.layers.experimental.preprocessing.Normalization(axis=-1)
# Apply the normalizer to the data:
normalizer.adapt(np.array(train_features))
# Calculate the mean and variance, and store them in the layer:
print(normalizer.mean.numpy())

# See the normalization of the first sample:
print(train_features.head())
first = np.array(train_features[:1])
with np.printoptions(precision=2, suppress=True):
    print("First example:", first)
    print("Normalized:", normalizer(first).numpy())

# ---------------------------------------------------- #
# Linear regression to predict "MPG" from "Horsepower" #
# ---------------------------------------------------- #

# Create a NumPy array made of the "Horsepower" feature. Then, instantiate the tf.keras.layers.Normalization
# and fit its state to the horsepower data:
horsepower = np.array(train_features["Horsepower"])

horsepower_normalizer = tf.keras.layers.experimental.preprocessing.Normalization(input_shape=[1, ], axis=None)
horsepower_normalizer.adapt(horsepower)

# Build the keras sequantial model:
horsepower_model = tf.keras.Sequential([
    horsepower_normalizer,
    layers.Dense(units=1)
])
# Line model summary:
horsepower_model.summary()

# Predicting MPG from Horsepower.
# Run the untrained model on the first 10 Horsepower values:
print(horsepower_model.predict(horsepower[:10]))

# Configure the trainning procedure:
horsepower_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss="mean_absolute_error")

# Use Keras Model.fit to execute the training for 100 epochs:
history = horsepower_model.fit(
    train_features["Horsepower"],
    train_labels,
    epochs=100,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split=0.2)

# Visualize the model's training progress using the stats
# stored in the history object:
hist = pd.DataFrame(history.history)
hist["epoch"] = history.epoch
print(hist.tail())

# Plot the training-testing accuracy/loss:
plot_loss(history)

# Check out the linear fit,
# input is Horsepower, output is MPG.

# Vector of Horsepower values from [0, 250]:
x = tf.linspace(0.0, 250, 251)
# MPG predictions for those Horsepower values:
y = horsepower_model.predict(x)

plot_horsepower(x, y)

# ------------------------------------------------------------------ #
# Linear regression to predict "MPG" from all the remaining features #
# ------------------------------------------------------------------ #

# Create a two-step Keras Sequential model again with the first layer being the
# normalizer that was defined earlier. Adapt it to the complete dataset:
linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])

# Line model summary:
linear_model.summary()

# Configure the trainning procedure:
linear_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

# Use Keras Model.fit to execute the training for 100 epochs:
history = linear_model.fit(
    train_features,
    train_labels,
    epochs=100,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split=0.2)

# Visualize the model's training progress using the stats
# stored in the history object:
hist = pd.DataFrame(history.history)
hist["epoch"] = history.epoch
print(hist.tail())

# Using all the inputs in this regression model should achieve
# a much lower training and validation error than the horsepower_model,
# which had one input:
plot_loss(history)

# Check out the predicted values vs the real ones to see the
# accuracy of the linear fit:
test_predictions = linear_model.predict(test_features).flatten()
plot_linearRegression(test_labels, test_predictions)

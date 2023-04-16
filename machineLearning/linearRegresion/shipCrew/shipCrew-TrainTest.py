# File        :   shipCrew-TrainTest.py
# Version     :   1.0.0
# Description :   [Train + Test]
#                 Linear reggression example for Predicting a Ship's Crew Size

# Date:       :   Apr 15, 2023
# Author      :   Ricardo Acevedo-Avila
# License     :   Creative Commons CC0

import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

from sklearn.svm import SVR
from sklearn.linear_model import Lasso

from sklearn.model_selection import cross_val_score
import seaborn as sns

# Project Path:
projectPath = "D://dataSets//ships//"

# File Names:
datasetName = "cruise_ship_info.csv"

# Cross-validation folds:
cvFolds = 5
# Cross-validation parallel jobs (1 per core):
parallelJobs = 5

# Read the CSV Dataset:
inputDataset = pd.read_csv(projectPath + datasetName)

# Set predicting feature (label/target):
label = "crew"

# Configure the console output:
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

# Check out the first five samples:
print("[INFO] --- Dataset 5 First Samples:")
print("Dataset shape: ", inputDataset.shape)
print(inputDataset.head())
print(" ")

# Check out some basic dataset stats:
print("[INFO] --- Dataset Stats:")
print(inputDataset.describe())
print(" ")

# Check out columns with missing data:
print("[INFO] --- Columns with missing data:")
print(inputDataset.isnull().sum())
print(" ")

# Important features selection
# Plot a person correlation heatmap to
# locate the top features that are hihgly correlated
# with the target feature (crew):

plt.figure(figsize=(6, 5))
# Get correlation matrix:
correlationMatrix = inputDataset.corr()
# Set plot of correlation matrix with color heatmap:
sns.heatmap(correlationMatrix, annot=True, cmap=plt.cm.Reds)
plt.show()

# Filter only the top correlated features:
correlatedThreshold = 0.85
# Get the candidate features that correlate to the target feature:
condidateFeatures = correlationMatrix[label]
# Apply the filter:
importantFeatures = condidateFeatures[condidateFeatures >= correlatedThreshold]
print("[INFO] --- These are the most positevely correlated features.")
print("Predicting feature is: " + str(label))
print(importantFeatures)
print(" ")

# Drop the target feature from important features, get feature names:
importantFeatures = importantFeatures.drop(label)
# Get feature names as a list:
importantFeatures = list(importantFeatures.index)

# Extract the target feature (label):
predictingFeature = pd.DataFrame({label: inputDataset[label]})

# Discard all features that are not important, keep only the
# relevant features:
inputDataset = inputDataset[importantFeatures]

# Dataset division, typically is training: 80% testing: 20%
print("[INFO] --- Splitting Complete Dataset...")
trainFeatures, testFeatures, trainLabel, testLabel = train_test_split(inputDataset, predictingFeature, test_size=0.40,
                                                                      random_state=0)
# Feature normalization
print("[INFO] --- Normalizing features...")
# Normalize all feature to same scale via z = (x - u) / s:
featureScaler = StandardScaler()

# Fit & transform train data:
trainFeatures = featureScaler.fit_transform(trainFeatures)
print(" Train data. Mean: " + str(trainFeatures.mean(axis=0)) + " Std. Dev: " + str(trainFeatures.std(axis=0)))

# Transform test data:
testFeatures = featureScaler.transform(testFeatures)
print(" Test data. Mean: " + str(testFeatures.mean(axis=0)) + " Std. Dev: " + str(testFeatures.std(axis=0)))

# Apply PCA:
reducedComponents = 4
print("[INFO] --- Applying PCA. Reducing components to: " + str(reducedComponents))

pca = PCA(n_components=reducedComponents)

print(" Train - Shape before PCA: ", trainFeatures.shape)
trainFeatures = pca.fit_transform(trainFeatures)
print(" Train - Shape after PCA: ", trainFeatures.shape)

print(" Test - Shape before PCA: ", testFeatures.shape)
testFeatures = pca.transform(testFeatures)
print(" Test - Shape after PCA: ", testFeatures.shape)

print(" PCA Variance Ratio: ", np.sum(pca.explained_variance_ratio_))

# Feed data to regressor:
print("[INFO] --- Creating and Fitting Linear Reggressor...")
linearRegressor = LinearRegression()

# Some alternate reggressors:
# linearRegressor = Lasso(alpha=0.1)
# linearRegressor = SVR(C=8.0, kernel="linear")

linearRegressor.fit(trainFeatures, trainLabel)

print("[INFO] --- Creating and Fitting Linear Reggressor...")
print(" Regressor coefficents: ")
print(linearRegressor.coef_)
print(" Regressor intercept: ")
print(linearRegressor.intercept_)

# Check out the reggressor accuracy using cross-validation:
print("[INFO] --- Cross-Validating Linear Reggressor...")
cvFolds = 5
reggressorAccuracy = cross_val_score(estimator=linearRegressor, X=trainFeatures, y=trainLabel, cv=cvFolds,
                                     n_jobs=parallelJobs, verbose=3)
# Accuracy for each fold:
print("[INFO] --- Fold Accuracy:")
print(" ", reggressorAccuracy)
print("[INFO] --- Mean & Std Dev Fold Accuracy:")
print(" Mu: ", np.mean(np.array(reggressorAccuracy)), "Sigma:", np.std(np.array(reggressorAccuracy)))

# Test the Linear Model:
print("[INFO] --- Testing Linear Reggressor...")
reggressorPredictions = linearRegressor.predict(testFeatures)
print(" Reggressor Test Accuracy:")
print(" ", linearRegressor.score(testFeatures, testLabel))

# Error plot. Check out the difference between the real and predicted values
# for both train and test:
trainPredictions = linearRegressor.predict(trainFeatures)

plt.scatter(trainPredictions.reshape(-1, 1), trainPredictions.reshape(-1, 1) - trainLabel,
            c='steelblue', marker='o', edgecolor='white', label="Training data")
plt.scatter(reggressorPredictions.reshape(-1, 1), reggressorPredictions.reshape(-1, 1) - testLabel,
            c='limegreen', marker='s', edgecolor='white', label="Test data")

plt.xlabel("Predicted values")
plt.ylabel("Prediction Error")

plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])

plt.tight_layout()
plt.legend(loc='lower right')

plt.show()

# Perform regression on a new sample:
print("[INFO] --- Predicting New Sample...")

# Set features:
newSample = [70.654, 10.3652, 5.01, 11.5123]
newSample = pd.DataFrame([newSample])

# Set feature names:
newSample.columns = ["Tonnage", "passengers", "length", "cabins"]

# Normalize new sample:
newSample = featureScaler.transform(newSample)

# Apply PCA:
newSample = pca.transform(newSample)

# Get new class/label/value:
newSampleClass = linearRegressor.predict(newSample)
print(" According to the Regressor, the new label is: " + str(newSampleClass[0]))

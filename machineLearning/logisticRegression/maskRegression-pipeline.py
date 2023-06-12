# File        :   maskRegression-pipeline.py
# Version     :   1.2.0
# Description :   [Train + Test]
#                 Logistic regression example for "Mask Data" custom dataset, using pipelines.

# Date:       :   May 22, 2023
# Author      :   Ricardo Acevedo-Avila
# License     :   Creative Commons CC0

import numpy as np
import pandas as pd
import random

from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


def showConfusionMatrix(yTest, modelPredictions, classes):
    confusionMatrix = confusion_matrix(yTest, modelPredictions, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=classes)
    disp.plot()
    plt.show()


# Project Path:
projectPath = "D://dataSets//"
# File Names:
datasetName = "maskData.csv"
# Prediction label:
predictionLabel = "classType"

# Read the dataset:
inputDataset = pd.read_csv(projectPath + datasetName)

# Check out the first 10 samples:
print(inputDataset.head(10))

# Select a fraction of rows:
sampledRows = inputDataset.sample(frac=0.10, random_state=42)

# "Pre-process" rows...
# Loop through the rows:
for i, row in sampledRows.iterrows():
    # Row to list:
    tempList = row.values.flatten().tolist()
    # Get total colums:
    totalColumns = len(row)
    # Get random column from 1 to 28:
    randomColumn = random.randint(12, totalColumns - 1)
    # Set random column to NaN:
    tempList[randomColumn] = np.nan
    # Concat to out dataframe:
    tempRow = pd.DataFrame([tempList], columns=sampledRows.columns, index=[i])

    # Into the sampledRows dataframe:
    inputDataset.loc[[i]] = tempRow

print(inputDataset.head(10))

# Free resources:
del sampledRows

# Check out class distribution:
classLabels = inputDataset[predictionLabel]
classCounter = Counter(classLabels)

# Print the class distribution:
for c in classCounter:
    # Get counter value
    value = classCounter[c]
    # Print distribution:
    print("Class:", c, "count:", value)

print("Total Samples", inputDataset.shape[0])

# Build the pipeline:
modelPipeline = Pipeline([("sampler", RandomOverSampler(sampling_strategy="minority", random_state=42)),
                          ("imputer", SimpleImputer(strategy="median")),
                          ("normalizer", StandardScaler()),
                          ("svm", svm.SVC(C=1.0, kernel="linear"))])

# Split the dataset:
inputLabels = inputDataset[predictionLabel]
inputDataset = inputDataset.drop(predictionLabel, axis=1)
xTrain, xTest, yTrain, yTest = train_test_split(inputDataset, inputLabels, test_size=0.2, random_state=42)

# Fit the pipeline:
modelPipeline.fit(xTrain, yTrain)

# Apply cross-validation:
crossScore = cross_val_score(estimator=modelPipeline, X=xTrain, y=yTrain, cv=5,
                             n_jobs=5, verbose=3)

# Accuracy for each fold:
print("Cross Val Score:", crossScore)
print("Mu: ", f'{np.mean(np.array(crossScore)):.4f}', "Sigma:", f'{np.std(np.array(crossScore)):.4f}')

# Score:
modelScore = modelPipeline.score(xTest, yTest)
print("Test Score", modelScore)

# Get the test predictions
modelPredictions = modelPipeline.predict(xTest)
testLabels = yTest.values.flatten().tolist()

# Check out the predictions:
for i in range(len(modelPredictions)):
    # Get prediction:
    predictedClass = modelPredictions[i]
    # Get label
    realClass = testLabels[i]
    mismatch = ""
    if predictedClass != realClass:
        mismatch = "<-"
    print("Sample:", i, "Predicted:", predictedClass, "Truth:", realClass, "" + mismatch)

# Show CF:
showConfusionMatrix(yTest, modelPredictions, modelPipeline["svm"].classes_)

# Show classification report:
print(classification_report(yTest, modelPredictions))

print("Running Search Grid...")
# Hyperparameter optimization using Grid Search:
paramGrid = {"svm__C": np.logspace(-0.1, 5, 10), "svm__kernel": ["linear", "poly", "rbf", "sigmoid"],
             "svm__gamma": np.logspace(0.1, 1, 10)}
gridResults = GridSearchCV(modelPipeline, paramGrid, n_jobs=2)

# Fit the pipeline:
gridResults.fit(xTrain, yTrain)
# Print hyperparameters:
print(gridResults.best_params_)

# Apply cross-validation:
crossScore = cross_val_score(estimator=gridResults, X=xTrain, y=yTrain, cv=5,
                             n_jobs=5, verbose=3)

# Accuracy for each fold:
print("Cross Val:", crossScore)
print("Mu: ", f'{np.mean(np.array(crossScore)):.4f}', "Sigma:", f'{np.std(np.array(crossScore)):.4f}')

# Score:
modelScore = gridResults.score(xTest, yTest)
print("Test SCore:", modelScore)

# Show CF:
modelPredictions = gridResults.predict(xTest)
showConfusionMatrix(yTest, modelPredictions, modelPipeline["svm"].classes_)

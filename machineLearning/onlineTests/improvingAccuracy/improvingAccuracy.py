# File        :   improvingAccuracy.py
# Version     :   1.5.0
# Description :   Solution to one of c0d1l1ty's Data Science "test"
#
# Date:       :   Apr 26, 2023
# Author      :   Mr. X
# License     :   Creative Commons CC0

# Instructions:
# Using only preprocessing, improve the classifier's accuracy.
# Your model will be scored using scikit-learn's roc_auc_score
# The data contains 7500 samples with 300 features. The feature
# names are not important. There are no missing values.

import numpy as np
import pandas as pd

from sklearn import metrics

from imblearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score

from collections import Counter
from sklearn.utils import resample


class ClassifierWithImbalanceClass:
    def __init__(self):
        # The pipeline:
        # PolynomialFeatures to raise existing features to an exponent (in this example, degree=2)
        # Typically linear algorithms, such as linear regression and logistic regression, respond well
        # to the use of polynomial input variables.
        # StandardScaler to normalize features
        self._pipeline = make_pipeline(PolynomialFeatures(degree=2), StandardScaler(),
                                       LinearSVC(C=1, random_state=42, dual=False))

    def train(self, x, y):
        # Fit the pipeline:
        self._pipeline.fit(x, y)
        pass

    def predict(self, x):
        # Return the test predictions:
        classPredictions = self._pipeline.predict(x)
        return classPredictions

    def getPipeline(self):
        # Returns the pipeline, so I can
        # run cross validation on it:
        return self._pipeline


# Project Path:
projectPath = "D://dataSets//code//"

# File Names:
trainDataset = "train_data.csv"
testDataset = "test_data.csv"

# Prediction label:
predictionLabel = "classType"

# Read the CSV Dataset:
df_train = pd.read_csv(projectPath + trainDataset)
y_train = df_train["target"]

# Check out the class distribution
classLabels = df_train["target"]
classCounter = Counter(classLabels)

# Print the class distribution:
for c in classCounter:
    value = classCounter[c]
    print("[Unbalanced] Class: ", c, " Count: ", value)

# Set count of majority class:
# Majority class is 0
maxClassCount = int(classCounter[0] * 1.0)

# Set the minority class to a separate dataframe:
dfMinority = df_train[df_train["target"] == 1]
# Set other classes to another dataframe:
dfMajority = df_train[df_train["target"] != 1]

# Up-sample the minority class:
minorityUpscaled = resample(dfMinority, random_state=42, n_samples=maxClassCount, replace=True)

# Concatenate the up-sampled dataframe. Now
# the dataset should be balanced:
df_train = pd.concat([minorityUpscaled, dfMajority])
print(df_train.shape)

# Check out the new class distribution
classLabels = df_train["target"]
classCounter = Counter(classLabels)

# Print the class distribution:
for c in classCounter:
    value = classCounter[c]
    print("[Balanced] Class: ", c, " Count: ", value)

# Get training features and label:
y_train = df_train["target"]
X_train = df_train.drop("target", axis=1)

# Read the test dataset:
df_test = pd.read_csv(projectPath + testDataset)
y_test = df_test["target"]
X_test = df_test.drop("target", axis=1)

# Call and implement classifier:
classifier = ClassifierWithImbalanceClass()
classifier.train(X_train, y_train)
y_pred = classifier.predict(X_test)

# Check out some metrics:
print(y_pred)
print(type(y_pred))
print(y_pred.shape)

auc = metrics.roc_auc_score(y_test, y_pred)
print("ROC:", auc)

# Get the pipeline:
modelPipeline = classifier.getPipeline()
# Check its score:
print("Pipe Score:", modelPipeline.score(X_test, y_test))

# Cross-validation folds:
cvFolds = 5
# Cross-validation parallel jobs (1 per core):
parallelJobs = 5

# Check out the classifier accuracy using cross-validation:
classifierAccuracy = cross_val_score(estimator=modelPipeline, X=X_train, y=y_train, cv=cvFolds,
                                     n_jobs=parallelJobs, verbose=3)

# Accuracy for each fold:
print("Cross Score:", classifierAccuracy)
# Mean accuracy and accuracy std dev:
print("Accuracy Mean: ", np.mean(np.array(classifierAccuracy)), "Std Dev:", np.std(np.array(classifierAccuracy)))

# Let's see the classification results:
for i in range(len(y_pred)):
    realClass = y_test[i]
    sampleClass = y_pred[i]

    # Mark mismatches with an arrow:
    missmatch = ""
    if realClass != sampleClass:
        missmatch = " <-"
    print("Sample:", i, "Predicted:", sampleClass, "Truth:", realClass, "" + missmatch)

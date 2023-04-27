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


class ClassifierWithImbalanceClass:
    def __init__(self):
        # The pipeline:
        # PolynomialFeatures to raise existing features to an exponent (in this example, degree=2)
        # Typically linear algorithms, such as linear regression and logistic regression, respond well
        # to the use of polynomial input variables.
        # StandardScaler to normalize features
        self._pipeline = make_pipeline(PolynomialFeatures(degree=2), StandardScaler(),
                                       LinearSVC(C=1, random_state=42, dual=True, max_iter=10000))

    def train(self, x, y):
        # Fit the pipeline:
        self._pipeline.fit(x, y)
        pass

    def predict(self, x):
        # Return the test predictions:
        classPredictions = self._pipeline.predict(x)
        return classPredictions

    def getClassifier(self):
        # Returns the classifier, so I can
        # run cross validation on it:
        return self._pipeline[2]


# Project Path:
projectPath = "D://dataSets//code//"

# File Names:
trainDataset = "train_data.csv"
testDataset = "test_data.csv"

# Read the CSV Dataset:
df_train = pd.read_csv(projectPath + trainDataset)
y_train = df_train["target"]

# Get correlation matrix:
correlationMatrix = df_train.corr()

# Discard negative-correlated features:
# Threshold:
correlatedThreshold = 0.0
# Get the candidate features that correlate to the target feature:
candidateFeatures = correlationMatrix["target"]

# Apply the filter:
importantFeatures = candidateFeatures[candidateFeatures >= correlatedThreshold]
importantFeatures = importantFeatures.drop("target")
# Get feature names as a list:
importantFeatures = list(importantFeatures.index)

# Discard all features that are not important, keep only the
# relevant features:
X_train = df_train[importantFeatures]

df_test = pd.read_csv(projectPath + testDataset)
y_test = df_test["target"]
X_test = df_test[importantFeatures]

# Call and implement classifier:
classifier = ClassifierWithImbalanceClass()
classifier.train(X_train, y_train)
y_pred = classifier.predict(X_test)

# Check out some metrics:
print(y_pred)
print(type(y_pred))
print(y_pred.shape)
auc = metrics.roc_auc_score(y_test, y_pred)
print(auc)

# Cross-validation folds:
cvFolds = 5
# Cross-validation parallel jobs (1 per core):
parallelJobs = 5

# Check out the reggressor accuracy using cross-validation:
classifierAccuracy = cross_val_score(estimator=classifier.getClassifier(), X=X_train, y=y_train, cv=cvFolds,
                                     n_jobs=parallelJobs,
                                     verbose=3)

# Accuracy for each fold:
print(classifierAccuracy)
# Mean accuracy and accuracy std dev:
print("Accuracy Mean: ", np.mean(np.array(classifierAccuracy)), "Std Dev:", np.std(np.array(classifierAccuracy)))

# Let's see the classification results:
# for i in range(len(y_pred)):
#     print(y_pred[i], y_test[i])

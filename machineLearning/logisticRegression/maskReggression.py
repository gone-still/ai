# File        :   maskReggression.py
# Version     :   1.0.0
# Description :   [Train + Test]
#                 Logistic reggression example for "Mask Data" custom dataset.

# Date:       :   Apr 18, 2023
# Author      :   Ricardo Acevedo-Avila
# License     :   Creative Commons CC0

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt
import seaborn as sns

# Project Path:
projectPath = "D://dataSets//"

# File Names:
datasetName = "maskData.csv"

# Prediction label:
predictionLabel = "classType"

# Cross-validation folds:
cvFolds = 5
# Cross-validation parallel jobs (1 per core):
parallelJobs = 5

# Read the CSV Dataset:
inputDataset = pd.read_csv(projectPath + datasetName)

# Check out the first five samples:
print("[INFO] --- Dataset 5 First Samples:")
print("Dataset shape: ", inputDataset.shape)
print(inputDataset.head())
print(" ")

# Configure the console output:
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
# pd.set_option("display.width", 1000)

# Check out some basic dataset stats:
print("[INFO] --- Dataset Stats:")
print(inputDataset.describe())
print(" ")

# Check out columns with missing data:
print("[INFO] --- Columns with missing data:")
print(inputDataset.isnull().sum())
print(" ")

# Create the target variable dictionary:
classDictionary = {"Real": 1, "Copy": 0}
inputDataset = inputDataset.replace({"classType": classDictionary})

# Important features selection
# Plot a person correlation heatmap to
# locate the top features that are highly correlated
# with the target feature (crew):

plt.figure(figsize=(5, 5))
# Get correlation matrix:
correlationMatrix = inputDataset.corr()
# Set plot of correlation matrix with color heatmap:
sns.heatmap(correlationMatrix, annot=True, cmap=plt.cm.Reds)
plt.show()

# Discard features that are below a correlation
# Threshold:
correlatedThreshold = 0.30
# Get the candidate features that correlate to the target feature:
condidateFeatures = correlationMatrix[predictionLabel]
# Apply the filter:
importantFeatures = condidateFeatures[condidateFeatures >= correlatedThreshold]
print("[INFO] --- These are the most positively correlated features.")
print("Predicting feature is: " + str(predictionLabel))
print(importantFeatures)
print(" ")

# Drop the target feature from important features, get feature names:
importantFeatures = importantFeatures.drop(predictionLabel)
# Get feature names as a list:
importantFeatures = list(importantFeatures.index)

# Extract the target feature (label):
predictingFeature = pd.DataFrame({predictionLabel: inputDataset[predictionLabel]})

# Discard all features that are not important, keep only the
# relevant features:
inputDataset = inputDataset[importantFeatures]

# Dataset split: 80% Training and 20% Testing
print("[INFO] --- Splitting Complete Dataset...")
# Get -> Training Features + Training Labels
#     -> Testing Features + Test Labels
trainFeatures, testFeatures, trainLabels, testLabels = train_test_split(inputDataset, predictingFeature, test_size=0.30,
                                                                        random_state=0)

# Feature normalization
print("[INFO] --- Normalizing features...")
# Normalize all feature to same scale via z = (x - u) / s:
featureScaler = StandardScaler()

# Normalize training features:
trainFeatures = featureScaler.fit_transform(trainFeatures)
# Normalize testing features:
testFeatures = featureScaler.transform(testFeatures)

# Create the logistic reggresor model:
logisticReggressor = LogisticRegression(solver="liblinear", C=1.0, random_state=0)

# Fit the model to the training data:
trainLabels = trainLabels.values.ravel()  # From column to row
logisticReggressor.fit(trainFeatures, trainLabels)

# Check out the reggressor accuracy using cross-validation:
print("[INFO] --- Cross-Validating Logistic Reggressor...")
reggressorAccuracy = cross_val_score(estimator=logisticReggressor, X=trainFeatures, y=trainLabels, cv=cvFolds,
                                     n_jobs=parallelJobs, verbose=3)

# Accuracy for each fold:
print("[INFO] --- Fold Accuracy:")
print(" ", reggressorAccuracy)
print("[INFO] --- Mean & Std Dev Fold Accuracy:")
print(" Mu: ", np.mean(np.array(reggressorAccuracy)), "Sigma:", np.std(np.array(reggressorAccuracy)))

# Test the Regression Model:
print("[INFO] --- Testing Logistic Reggressor...")
predictionProbabilities = logisticReggressor.predict_proba(testFeatures)
regressorPredictions = logisticReggressor.predict(testFeatures)

testLabels = testLabels.values.ravel()  # From column to row
regressorScore = logisticReggressor.score(testFeatures, testLabels)
print(" Accuracy:", regressorScore)

print("[INFO] --- Per sample accuracy...")
# Print the predicted class, real class and probabilities per sample:
for i in range(len(testFeatures)):
    # Get sample max probability:
    sampleProbability = np.max(predictionProbabilities[i])
    # Get predicted class:
    sampleClass = np.argmax(predictionProbabilities[i])
    # Get real class:
    realClass = testLabels[i]
    # Print missmatch:
    missmatch = ""
    if realClass != sampleClass:
        missmatch = " <-"
    # Print the info:
    print(" Sample:", i, "Predicted:", sampleClass, "Truth:", realClass,
          "(Proba: " + "{:.4f}".format(sampleProbability) + ")"+missmatch)

# Get confusion matrix and its plot:
print("[INFO] --- Plotting CM")
confusionMatrix = confusion_matrix(testLabels, regressorPredictions, labels=logisticReggressor.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=logisticReggressor.classes_)
disp.plot()
plt.show()

# Plot ROC:
fpr, tpr, proba = metrics.roc_curve(testLabels, predictionProbabilities[::, 1])
auc = metrics.roc_auc_score(testLabels, predictionProbabilities[::, 1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
plt.legend(loc=4)
plt.show()

# Optimal probability threshold according to the ROC curve
# Both classes unweighted. If proba >= 1 -> outClass = 1, else: outClass = 0
probabilityThreshold = sorted(list(zip(np.abs(tpr - fpr), proba)), key=lambda i: i[0], reverse=True)[0][1]
print("[INFO] --- Probability Threshold: " + "{:.4f}".format(probabilityThreshold))

# Print Accuracy report:
accuracyReport = classification_report(testLabels, regressorPredictions)
print("[INFO] --- Accuracy Report:", accuracyReport, sep="\n", end="\n\n")

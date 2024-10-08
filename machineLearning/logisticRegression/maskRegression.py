# File        :   maskRegression.py
# Version     :   1.2.0
# Description :   [Train + Test]
#                 Logistic regression example for "Mask Data" custom dataset.

# Date:       :   May 20, 2023
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
from sklearn.metrics import precision_recall_curve

from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import GridSearchCV
import seaborn as sns

# The TK gui crashes while debugging with opened windows,
# Change graph backend to TkAgg:
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use("TkAgg")


# Prints the class distribution of
# a dataset:
def checkClassDistribution(dataset, targetFeature):
    classLabels = dataset[targetFeature]
    classCounter = Counter(classLabels)

    # Print the class distribution:
    for c in classCounter:
        value = classCounter[c]
        print("Class: ", c, " Count: ", value)

    print("Total Samples: ", dataset.shape)


# Project Path:
projectPath = "D://dataSets//"

# File Names:
datasetName = "maskData.csv"

# Prediction label:
predictionLabel = "classType"

# Cross-validation folds:
cvFolds = 10
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
correlatedThreshold = 0.20
# Get the candidate features that correlate to the target feature:
condidateFeatures = correlationMatrix[predictionLabel]
# Apply the filter:
importantFeatures = condidateFeatures[abs(condidateFeatures) >= correlatedThreshold]
print("[INFO] --- These are the most positively correlated features.")
print("Predicting feature is: " + str(predictionLabel))

# Get features names as a list:
importantFeaturesNames = list(importantFeatures.index)
# Print the filtered features:
for i in range(len(importantFeatures)):
    # Print the feature index, feature name and its correlation value (to 4 decimal places):
    print(i, ":", importantFeaturesNames[i], ",", "{:.4f}".format(importantFeatures[i]))
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
trainFeatures, testFeatures, trainLabels, testLabels = train_test_split(inputDataset, predictingFeature, test_size=0.25,
                                                                        random_state=42)

# Create trainning dataset:
trainFeatures = trainFeatures.reset_index(drop=True)
trainLabels = trainLabels.reset_index(drop=True)
trainingDataset = trainFeatures.join(trainLabels)

# Check out the class distribution:
print("[INFO] --- Train Dataset class distribution [Pre-sampling]:")
checkClassDistribution(trainingDataset, predictionLabel)

# Resample to balance out both classes:
ros = RandomOverSampler(sampling_strategy="minority", random_state=42)
trainFeatures, trainLabels = ros.fit_resample(trainFeatures, trainLabels)

# Create the full, oversampled, dataset:
trainingDataset = trainFeatures.join(trainLabels)

# Check out the new class distribution:
print("[INFO] --- Train Dataset class distribution [Post-sampling]:")
checkClassDistribution(trainingDataset, predictionLabel)

# Feature normalization
print("[INFO] --- Normalizing features...")
# Normalize all feature to same scale via z = (x - u) / s:
featureScaler = StandardScaler()

# Normalize training features:
trainFeatures = featureScaler.fit_transform(trainFeatures)
# Normalize testing features:
testFeatures = featureScaler.transform(testFeatures)

# Create the logistic reggresor model:
logisticReggressor = LogisticRegression(solver="liblinear", C=1.0, random_state=42)

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
          "(Proba: " + "{:.4f}".format(sampleProbability) + ")" + missmatch)

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
# Both classes unweighted. If proba >= probabilityThreshold -> outClass = 1, else: outClass = 0
probabilityThreshold = sorted(list(zip(np.abs(tpr - fpr), proba)), key=lambda i: i[0], reverse=True)[0][1]
print("[INFO] --- Probability Threshold: " + "{:.4f}".format(probabilityThreshold))

# Print Accuracy report:
accuracyReport = classification_report(testLabels, regressorPredictions)
print("[INFO] --- Accuracy Report:", accuracyReport, sep="\n", end="\n\n")

# Hyperparameter optimization using Grid Search:
print("[INFO] --- Running Grid Search...")
parameters = {"solver": ["liblinear"], "C": np.logspace(-2, 2, 30), "penalty": ["l1", "l2"]}
logisticReggressor = LogisticRegression(max_iter=400)
logisticReggressorOptimized = GridSearchCV(logisticReggressor, parameters, cv=cvFolds, n_jobs=parallelJobs)
logisticReggressorOptimized.fit(trainFeatures, trainLabels)

# Print hyperparameters & accuracy:
print("[INFO] --- Grid Search Best Parameters:")
print("", logisticReggressorOptimized.best_params_)

# Check out the reggressor accuracy using cross-validation:
print("[INFO] --- [Post-Grid Search] Cross-Validating Logistic Reggressor...")
reggressorAccuracy = cross_val_score(estimator=logisticReggressorOptimized, X=trainFeatures, y=trainLabels, cv=cvFolds,
                                     n_jobs=parallelJobs, verbose=3)

# Accuracy for each fold:
print("[INFO] --- [Post-Grid Search] Fold Accuracy:")
print(" ", reggressorAccuracy)
print("[INFO] --- [Post-Grid Search] Mean & Std Dev Fold Accuracy:")
print(" Mu: ", np.mean(np.array(reggressorAccuracy)), "Sigma:", np.std(np.array(reggressorAccuracy)))

# Test the Regression Model:
print("[INFO] --- [Post-Grid Search] Testing Logistic Reggressor...")
predictionProbabilities = logisticReggressorOptimized.predict_proba(testFeatures)
regressorPredictions = logisticReggressorOptimized.predict(testFeatures)

regressorScore = logisticReggressorOptimized.score(testFeatures, testLabels)
print(" Accuracy:", regressorScore)

print("[INFO] --- [Post-Grid Search] Per sample accuracy...")

# Print the predicted class, real class and probabilities per sample:
for i in range(len(testFeatures)):
    # Get sample max probability:
    sampleProbability = np.max(predictionProbabilities[i])
    # Get predicted class:
    sampleClass = np.argmax(predictionProbabilities[i])
    # Get real class:
    realClass = testLabels[i]

    # Custom prediction set at a probability of 0.59,
    # to keep a reasonable precision/recall ratio for the application
    # (Trying to boost precision at the cost of recall).
    # 0.5688 which yields a ROC point of (0.7857, 0.1538):
    if sampleProbability > 0.5688:
        customClass = sampleClass
    else:
        customClass = 0

    # Print missmatch:
    missmatch = " "
    if realClass != sampleClass:
        missmatch = " <- "
    if realClass != customClass:
        missmatch = missmatch + "*"

    # Print the info:
    print(" Sample:", i, "Predicted:", sampleClass, "Truth:", realClass, "Cus:", customClass,
          "(Proba: " + "{:.4f}".format(sampleProbability) + ")" + missmatch)

# Get confusion matrix and its plot:
print("[INFO] --- [Post-Grid Search] Plotting CM")
confusionMatrix = confusion_matrix(testLabels, regressorPredictions, labels=logisticReggressorOptimized.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=logisticReggressorOptimized.classes_)
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
# Both classes unweighted. If proba >= probabilityThreshold -> outClass = 1, else: outClass = 0
probabilityThreshold = sorted(list(zip(np.abs(tpr - fpr), proba)), key=lambda i: i[0], reverse=True)[0][1]
print("[INFO] --- [Post-Grid Search] Probability Threshold: " + "{:.4f}".format(probabilityThreshold))

# Print Accuracy report:
accuracyReport = classification_report(testLabels, regressorPredictions)
print("[INFO] --- [Post-Grid Search] Accuracy Report:", accuracyReport, sep="\n", end="\n\n")

from sklearn.model_selection import cross_val_predict

y_scores = cross_val_predict(logisticReggressorOptimized, trainFeatures, trainLabels, cv=3,
                             method="decision_function")

# Plot precision/recall graph:
precisions, recalls, thresholds = precision_recall_curve(trainLabels, y_scores)
threshold = 0

plt.figure(figsize=(8, 4))  # extra code – it's not needed, just formatting
plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
plt.vlines(threshold, 0, 1.0, "k", "dotted", label="threshold")

# extra code – this section just beautifies and saves Figure 3–5
idx = (thresholds >= threshold).argmax()  # first index ≥ threshold
plt.plot(thresholds[idx], precisions[idx], "bo")
plt.plot(thresholds[idx], recalls[idx], "go")
plt.grid()
plt.xlabel("Threshold")
plt.legend(loc="center right")
plt.show()

plt.plot(recalls, precisions, linewidth=2, label="Precision/Recall curve")
plt.show()

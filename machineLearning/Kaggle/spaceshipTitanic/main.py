# File        :   spaceshipTitanic.py
# Version     :   2.1.0
# Description :   Solution for Kaggle"s Spaceship Titanic problem
#                 (https://www.kaggle.com/competitions/spaceship-titanic)

# Date:       :   Nov 20, 2023
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT

import numpy as np
import pandas as pd
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras import regularizers
from tensorflow.keras.models import load_model

from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint

import random

import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import IsolationForest

from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, AdaBoostClassifier

from sklearn.model_selection import cross_val_score

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, RandomizedSearchCV, HalvingGridSearchCV, \
    HalvingRandomSearchCV

from sklearn.base import clone

from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_score, \
    recall_score, precision_recall_curve, f1_score, accuracy_score

import time


# Replaces one "target" value in a column with a
# "value" one:
def replaceFeatureValue(datasetFeature, target, value):
    datasetFeature = datasetFeature.replace(target, value)
    return datasetFeature


# Processes "Age" feature:
def processAge(inputDataset, ageBins, binLabels):
    # Replace "NaN" with "-0.5":
    inputDataset["Age"] = replaceFeatureValue(inputDataset[["Age"]], np.NaN, -0.5)
    # Segment feature into bins:
    labeledAge = pd.cut(currentDataset["Age"], bins=ageBins, labels=binLabels)
    # Convert series to data frame:
    labeledAge = pd.DataFrame(labeledAge)

    return labeledAge


def economicStatus(person):
    # Creating new column

    # List of tests:
    peopleChoices = []

    # Test if these features are exactly == 0.0.
    # If feature == 0 -> True, else: False:
    peopleChoices.append(person.VIP == "VIP-FALSE")
    peopleChoices.append(person.RoomService == 0.0)
    peopleChoices.append(person.FoodCourt == 0.0)
    peopleChoices.append(person.ShoppingMall == 0.0)
    peopleChoices.append(person.Spa == 0.0)
    peopleChoices.append(person.VRDeck == 0.0)

    # Count number of "True"s in list:
    count = peopleChoices.count(True)

    # Bin the count:
    #     Count >= 5 -> 1
    # 5 > Count >= 3 -> 2
    # 3 > Count >= 1 -> 3
    # 1 > Count >= 0 -> 0
    if count >= 5:
        return 1
    elif count >= 3 and count < 5:
        return 2
    elif count >= 1 and count < 3:
        return 3
    else:
        return 0


def getTransportProbas(person):
    # CryoSleep:
    # CryoSleep -> False (Low prob of being transported)
    # CryoSleep -> True (High prob of being transported)

    # Home Planet:
    # Europa -> 1 (High prob of being transported)
    # Earth -> 1 (Low prob of being transported)

    # Age Group:
    # AgeGroup1-True -> 1 (High prob of being transported)
    # AgeGroup2-False -> 1 (Low prob of being transported)

    highProba = ["CryoSleep-TRUE", "Europa", "AgeGroup1-True"]
    lowProba = ["CryoSleep-FALSE", "Earth", "AgeGroup2-False"]
    featureList = [highProba, lowProba]

    # Stores feature count for high and low proba features:
    counters = [0, 0]

    # Count if there's a match:
    for f, currentFeatureList in enumerate(featureList):
        for currentFeature in currentFeatureList:
            currentValue = int(person[currentFeature])
            if currentValue == 1:
                counters[f] += 1
    # print(counters)
    return counters


def transportedLikelihood(person, encode=False):
    # featureList = ["HighProba", "LowProba"]

    # Get features
    highProba = person["HighProba"]
    lowProba = person["LowProba"]

    # Compute difference:
    featureDifference = highProba - lowProba
    outValue = featureDifference

    # Check if the feature needs encoding:
    if encode:
        # Feature is negative (Likely not transported):
        if featureDifference < 0:
            outValue = -1
        else:
            # Feature is 0 (Can't tell if transported or not):
            if featureDifference == 0:
                outValue = 0
            # Feature is positive (Likely transported):
            else:
                outValue = 1

    return outValue


# Performs stratified cross-validation:
def stratifiedCrossValidation(currentModel, trainFeatures, trainLabels, randomState, splits=10, testSize=0.2,
                              runCV=True, verbose=False):
    if not runCV:
        print("Skipping stratified crossvalidation...")
        return [], 0.0, 0.0
    # Get the stratified partitioned object from dataset:
    cvFolds = StratifiedShuffleSplit(n_splits=splits, test_size=testSize, random_state=randomState)
    # indices = cvFolds.get_n_splits(trainFeatures, trainLabels)

    # This list stores all fold accuracies:
    accuracyPerFold = []

    # Loop through the folds, extracting features and labels from the split object:
    for i, (train_index, test_index) in enumerate(cvFolds.split(trainFeatures, trainLabels)):
        # Get train fold:
        foldTrainFeatures = trainFeatures.iloc[train_index, :]
        foldTrainLabels = trainLabels[train_index]
        # showClassDistribution(foldTrainLabels)

        # Get test fold:
        foldTestFeatures = trainFeatures.iloc[test_index, :]
        foldTestLabels = trainLabels[test_index]
        # showClassDistribution(foldTestLabels)

        # Clone model:
        modelClone = clone(currentModel)
        # Fit:
        modelClone.fit(foldTrainFeatures, foldTrainLabels)
        # Get predictions
        foldPredictions = modelClone.predict(foldTestFeatures)
        # Get number of correct predictions:
        correctPredictions = sum(foldPredictions == foldTestLabels)
        # Get accuracy:
        foldAccuracy = correctPredictions / len(foldPredictions)
        if verbose:
            print("Fold:", i, "Accuracy:", foldAccuracy)

        # Append to list:
        accuracyPerFold.append(foldAccuracy)

    # Get mean accuracy:
    meanAccuracy = np.mean(accuracyPerFold)
    # Get accuracy std dev across all folds:
    stdDevAccuracy = np.std(accuracyPerFold)
    print("Mean cross-validation accuracy: ", meanAccuracy, "stdDev: ", stdDevAccuracy)
    # Done:
    return accuracyPerFold, meanAccuracy, stdDevAccuracy


# Plots a confusion matrix:
def plotConfusionMatrix(testLabels, modelPredictions, model):
    # modelPredictions = model.predict(testFeatures)
    confusionMatrix = confusion_matrix(testLabels, modelPredictions, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=model.classes_)
    disp.plot()
    plt.show()


# Get per sample accuracy and returns counter of test (real) labels:
def perSampleAccuracy(modelPredictions, testLabels, verbose=True):
    # Real classes counter:
    classesCounter = {"0": 0, "1": 0}

    # List of accuracies per sample:
    # sampleAccuracies = []

    # Print the predicted class, real class and probabilities per sample:
    for i in range(len(testLabels)):

        # Get sample max probability:
        # sampleProbability = np.max(predictionProbabilities[i])
        # Into the list:
        # sampleAccuracies.append(sampleProbability)
        # Get predicted class:
        sampleClass = modelPredictions[i]
        # Get real class:
        realClass = testLabels[i]
        # Into class counter:
        classesCounter[str(realClass)] += 1
        # Print missmatch:
        missmatch = ""
        if realClass != sampleClass:
            missmatch = " <-"

        # Print the info:
        if verbose:
            print(" Sample:", i, "Truth:", realClass, "Predicted:", sampleClass, missmatch)

    return classesCounter


# Computes and prints results:
def displayResults(modelPredictions, testLabels, modelScore, cvMean, cvStdDev, cvFolds):
    # The out values:
    outValues = {}
    # Print the confusion matrix array:
    cmArray = confusion_matrix(testLabels, modelPredictions)  # normalize="pred"
    print("Confusion Matrix: ")
    print(cmArray)

    # Get accuracy from CM:
    accuracy = (cmArray[0][0] + cmArray[1][1]) / len(testLabels)
    # Into the out list:
    outValues["cmAccuracy"] = accuracy

    # Compute precision & recall:
    modelPrecision = precision_score(testLabels, modelPredictions)
    modelRecall = recall_score(testLabels, modelPredictions)

    # Compute F1 score:
    f1Score = f1_score(testLabels, modelPredictions)

    # Print the results:
    dateNow = time.strftime("%Y-%m-%d %H:%M")
    print("---------------------------------------------------------- ")
    print("Results Test time: " + dateNow)
    print("Precision: ", modelPrecision)
    print("Recall: ", modelRecall)
    print("F1: ", f1Score)
    print("---------------------------------------------------------- ")
    print("Validation CM Accuracy:", accuracy)
    print("Validation Accuracy [NT]:", modelScore)

    print(">> Cross-validation Mean (" + str(cvFolds) + " Folds): ", "{:.4f}".format(cvMean),
          "StdDev: ", "{:.4f}".format(cvStdDev))

    return outValues


# Shows the distribution of a two-classes dataset:
def showClassDistribution(classLabels):
    # Get class distribution:
    classCounter = Counter(classLabels)

    # Get total entries:
    totalEntries = len(classLabels)

    # Print the class distribution:
    for c in classCounter:
        # Get counter value
        value = classCounter[c]
        percent = format(100 * (value / totalEntries), ".2f")
        # Print distribution:
        print("Class:", c, "count:", value, "{ " + str(percent) + "%" + " }")


# Shows most and least important features if current model is Random Forest:
def showFeatures(currentModel, trainFeatures, maxSCoreThreshold=0.95):
    # Store the features here:
    featureList = []
    for score, name in zip(currentModel.feature_importances_, trainFeatures.columns):
        # Store the score along its feature name:
        featureList.append((score, name))

    # Sort from largest to smallest according to first tuple element:
    featureList.sort(key=lambda tup: tup[0], reverse=True)

    # Create lists for plotting:
    totalFeatures = len(featureList)
    # scoreSorted = []
    featuresSorted = [None] * totalFeatures

    # Set score total threshold:
    leastImportantFeatures = []
    mostImportantFeatures = []

    # Loop thru the features:
    scoreAccumulator = 0.0
    for f, currentTuple in enumerate(featureList):
        # Get current tuple:
        currentScore, currentFeature = currentTuple

        # Reverse-store score and feature name:
        # scoreSorted.append(currentScore)
        featuresSorted[(totalFeatures - 1) - f] = currentFeature

        # Accumulate score:
        if scoreAccumulator < maxSCoreThreshold:
            # Store most important feature:
            mostImportantFeatures.append(currentTuple)
            scoreAccumulator += currentScore
        else:
            # Store least important features:
            leastImportantFeatures.append(currentTuple)

        # Print info:
        print("{:.4f}".format(currentScore), "(" + "{:.4f}".format(scoreAccumulator * 100) + "%)", currentFeature)

    # Print the least important features:
    filterFeatures = []
    for currentTuple in leastImportantFeatures:
        # Get current tuple:
        currentScore, currentFeature = currentTuple
        filterFeatures.append(currentFeature)

    print("Least important features: ")
    print(filterFeatures)

    # Reverse lists using score:
    mostImportantFeatures.sort(key=lambda tup: tup[0], reverse=False)
    leastImportantFeatures.sort(key=lambda tup: tup[0], reverse=False)

    # Plot the features:
    plt.title("Feature importance", fontsize=10)
    plt.xlabel("Importance", fontsize=13)

    # Get lists of vertical units:
    lowRange = list(range(len(leastImportantFeatures)))
    highRange = list(range(len(mostImportantFeatures)))
    # Add offset to the bars:
    highRange = [x + len(leastImportantFeatures) for x in highRange]

    # Get just scores (first element of tuple) from feature lists:
    lowScores = [j[0] for j in leastImportantFeatures]
    highScores = [j[0] for j in mostImportantFeatures]

    # Plot two groups of horizontal bars -> least important (red), most important (green)
    plt.barh(lowRange, lowScores, color="red", edgecolor="red")
    plt.barh(highRange, highScores, color="green", edgecolor="green")

    # Add feature names to the vertical axis:
    plt.yticks(range(len(featuresSorted)), featuresSorted)
    plt.show()


def bestThreshold(y_test_list_norm, y_pred_proba_norm):
    scores = []
    thresholds = []

    best_score = 0
    best_threshold = 0

    for threshold in np.arange(0.4, 0.81, 0.01):
        predictions = (y_pred_proba_norm > threshold).astype(int)
        m = accuracy_score(y_test_list_norm, predictions)
        scores.append(m)
        thresholds.append(threshold)
        if m > best_score:
            best_score = m
            best_threshold = threshold
            # print(m)

    # print(scores)
    # print(thresholds)

    return best_threshold


# Standardizes a feature:
def standardizeFeature(featureName, dataset, datasetName, scalerType, encoderDictionary):
    print("-> Normalizing Feature: ", featureName, " Type: ", scalerType)

    # Get feature type (should be a str or a list):
    isString = isinstance(featureName, str)
    if not isString:
        dictName = '-'.join(featureName)
    else:
        dictName = featureName

    if datasetName == "train":

        print(">> [Train] Fitting + Transforming: ", featureName)

        # Set scaler:
        print("Using scaler -> ", scalerType)
        if scalerType == "MinMax":
            currentScaler = MinMaxScaler()
        elif scalerType == "Standard":
            currentScaler = StandardScaler()
        else:
            # Option not found:
            raise TypeError("standarizeNumerical>> Error: Received unsupported scaling type: " + scalerType)

        # Fit + transform:
        if isString:
            dataset[featureName] = currentScaler.fit_transform(dataset[[featureName]])
        else:
            dataset[featureName] = currentScaler.fit_transform(dataset[featureName])

        # Store transformer into dict:
        encoderDictionary[dictName + "-Scaler"] = currentScaler

    else:

        print(">> [Val/Test] Transforming: ", featureName)
        # Get Scaler:
        currentScaler = encoderDictionary[dictName + "-Scaler"]

        # Transform feature:
        if isString:
            dataset[featureName] = currentScaler.transform(dataset[[featureName]])
        else:
            dataset[featureName] = currentScaler.transform(dataset[featureName])


# Imputes missing data:
def imputeFeature(featureName, dataset, datasetName, imputerConfig, encoderDictionary):
    print("-> Imputing Feature: ", featureName, ", Imputer Config:", imputerConfig)

    # Get feature type (should be a str or a list):
    isString = isinstance(featureName, str)
    if not isString:
        dictName = '-'.join(featureName)
    else:
        dictName = featureName

    # Get imputer config:
    missingValue = imputerConfig["missingValue"]
    imputingStrategy = imputerConfig["strategy"]

    if datasetName == "train":

        print(">> [Train] Fitting + Transforming: ", featureName)

        # Set imputer:
        currentImputer = SimpleImputer(missing_values=missingValue, strategy=imputingStrategy)

        # Fit + transform feature:
        if isString:
            dataset[featureName] = currentImputer.fit_transform(dataset[[featureName]])
        else:
            dataset[featureName] = currentImputer.fit_transform(dataset[featureName])

        # Store transformer into dict:
        encoderDictionary[dictName + "-Imputer"] = currentImputer

    else:

        print(">> [Val/Test] Transforming: ", featureName)

        # Get imputer:
        currentImputer = encoderDictionary[dictName + "-Imputer"]

        # Transform feature:
        if isString:
            dataset[featureName] = currentImputer.transform(dataset[[featureName]])
        else:
            dataset[featureName] = currentImputer.transform(dataset[featureName])


# One-hot encode feature:
def oheFeature(featureName, inputDataset, outputDataset, datasetName, encoderDictionary):
    print("-> One-hot encoding feature:", featureName)

    if datasetName == "train":
        print(">> [Train] Fitting + Transforming: ", featureName)
        # Create the encoder object:
        currentEncoder = OneHotEncoder()
        # Fit + transform to feature:
        encodedFeature = currentEncoder.fit_transform(inputDataset[[featureName]])

        # Store encoder into dictionary:
        encoderDictionary[featureName + "-OHE"] = currentEncoder

    else:
        print(">> [Val/Test] Transforming: ", featureName)
        # Create the encoder object:
        currentEncoder = encoderDictionary[featureName + "-OHE"]
        # Transform the feature:
        encodedFeature = currentEncoder.transform(inputDataset[[featureName]])

    # Convert to array:
    encodedFeature = encodedFeature.toarray()

    # Print categories:
    print("OHE:" + currentFeature, "Categories: ", currentEncoder.categories_)

    # Append to output dataset:
    outputDataset[currentEncoder.categories_[0]] = encodedFeature

    # Return the new categories:
    return currentEncoder.categories_[0]


# Project Path:
projectPath = "D://dataSets//spaceTitanic//"
# Output Path:
outFilename = "outPredictions"
# Write final CSV?
writeOutfile = False

# File Names:
datasetNames = ["train", "validation", "test"]

# Dataset file extension:
fileExtension = ".csv"

# Prediction label:
predictionLabel = "Transported"

# Script options
randomSeed = 42

# Global rng object:
rng = np.random.RandomState(randomSeed)

# Normalize binary / categorical variables:
normalizeBinary = False
normalizeCategorical = True
categoricalScaler = "MinMax"
numericalScaler = "Standard"
binaryScaler = "Standard"

# Should original numerical features be attached to dataset?
includeNumerical = True

# Number of bin partitions of each numerical feature:
numericalBins = 4

# Test split:
# -1 uses all train dataset for training (no validation):
testSplit = 0.2

# Model type:
modelType = "VotingClassifier"
svmVoting = "hard"

# Run final dataset through best predictor:
getFinalPredictions = False

# Fit shallow predictors:
fitShallow = True

# Use dnn to predict a new feature?
dnnVote = False

# Should the DNN be run?
runDNN = False
# DNN epochs:
totalEpochs = 100
# Batch size:
dnnBatchSize = 32
# Learning rate:
dnnLearningRate = 0.0002

# Store best model here:
bestModel = {"Model": None, "Accuracy": 0.0, "Threshold": 0.0}

# Compute the best threshold?
computeThreshold = False

# Skip stratified CV:
skipStratifiedCV = True

# Cross-validation folds:
cvFolds = 10
runGridSearch = False
# Cross-validation parallel jobs (1 per core):
parallelJobs = 5

# Feature Selection:
showFeatureImportance = False
featureScoreThreshold = 0.98

# Drop features in list?
filterFeatures = False

# The list of features to drop:
# featuresList = ["C", "Infant", "Child", "NA-Age", "55 Cancri e", "NA-CryoSleep", "B", "TRAPPIST-1e", "Young Adult",
#                 "Adult", "D", "PSO J318.5-22", "VIP-FALSE", "Senior", "NA-Cabin-2", "NA-VIP", "NA-HomePlanet",
#                 "NA-Destination", "NA-Cabin-0", "VIP-TRUE", "Teenager", "A", "T"]

featuresList = ["T"]

# Set console format:
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

# Set random seed:
random.seed(randomSeed)

# Dictionary of prediction targets:
predictionTargets = {}

# Prepare the dictionary of encoders:
encodersDictionary = {}

# Prepare the datasets dictionary:
# train: {"Dataset", "Labels}
datasets = {"train": {},
            "validation": {},
            "test": {}}

# Store here the test passenger columns for final CSV:
passengerList = pd.DataFrame()

for d, datasetName in enumerate(datasetNames):

    print("Processing Dataset:", datasetName)

    # Select dataset type:
    if datasetName == "train":

        # Read the cvs file:
        currentDataset = pd.read_csv(projectPath + datasetName + fileExtension)

        # plt.figure(figsize=(5, 5))
        # sns.histplot(currentDataset, x='Age', hue='Transported', kde=True)
        # plt.show()

        # Create test split?
        if testSplit != -1:
            # Split the training dataset into train + validation:
            print("Splitting training + validation datasets...")
            trainDataset, validationDataset = train_test_split(currentDataset, test_size=testSplit, random_state=rng)

            # Store the validation dataset:
            datasets["validation"]["dataset"] = validationDataset

        else:
            # No test, use all train dataset to train the classifier:
            trainDataset = currentDataset

        # Set the dataset to be processed:
        currentDataset = trainDataset

    elif datasetName == "validation":

        # Check split:
        if testSplit == -1:
            print("No validation dataset provided, skipping...")
            continue

        print("Setting validation dataset")
        validationDataset = datasets["validation"]["dataset"]

        # Set the dataset to be processed:
        currentDataset = validationDataset

    else:

        print("Setting test dataset...")
        # Read the cvs file:
        testDataset = pd.read_csv(projectPath + datasetName + fileExtension)

        # Set the dataset to be processed:
        currentDataset = testDataset

    # Print dataset shape:
    print(currentDataset.shape)

    # Get a quick description of the data:
    print(currentDataset.info())

    # Check out numerical features only:
    print(currentDataset.describe())

    # Drop PassengerId and Name columns:
    dropFeatures = ["PassengerId", "Name"]
    for currentFeature in dropFeatures:
        if currentFeature == "PassengerId" and datasetName == "test":
            print("Storing: ", currentFeature)
            # Store passenger list for final predictions
            passengerList = pd.DataFrame(currentDataset["PassengerId"])

        print("Dropping column: ", currentFeature, "(" + str(datasetName) + ")")
        currentDataset = currentDataset.drop(currentFeature, axis=1)

    # Get feature names:
    featureNames = currentDataset.columns.values.tolist()

    # Process predictive feature:
    if datasetName == "train" or datasetName == "validation":
        # Replace "Transported" Feature with 0 (False) or 1 (True):
        print("Setting Predictive Feature...")
        currentDataset[predictionLabel] = replaceFeatureValue(currentDataset[predictionLabel], True, 1)
        currentDataset[predictionLabel] = replaceFeatureValue(currentDataset[predictionLabel], False, 0)

        # Get class labels:
        classLabels = currentDataset[predictionLabel]

        # Get the training features and target feature/predicting label:
        predictionTarget = pd.DataFrame(data=currentDataset[predictionLabel], columns=[predictionLabel]).reset_index(
            drop=True)
        currentDataset = currentDataset.drop(predictionLabel, axis=1)

        # Store predicting labels:
        datasets[datasetName]["labels"] = predictionTarget

        print("Computing Class Distribution...")

        showClassDistribution(classLabels)

    # NaN Replacement of the following features:
    targetFeatures = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
    for currentFeature in targetFeatures:
        # Replace "NaN" with "NA-FEATURE":
        replacementString = "NA-" + currentFeature
        print("Replacing NaNs in: ", currentFeature, "with", replacementString)

        currentDataset[currentFeature] = replaceFeatureValue(currentDataset[currentFeature], np.NaN,
                                                             replacementString)

        # Additionally, replace "True" and "False" in "CryoSleep/VIP" Column:
        if currentFeature == "CryoSleep" or currentFeature == "VIP":
            print("Replacing True/False in: ", currentFeature)
            currentDataset[currentFeature] = replaceFeatureValue(currentDataset[currentFeature], True,
                                                                 currentFeature + "-TRUE")
            currentDataset[currentFeature] = replaceFeatureValue(currentDataset[currentFeature], False,
                                                                 currentFeature + "-FALSE")
    # Prepare the final preprocessed dataset:
    preprocessedDataset = pd.DataFrame()

    # Economic status:
    featureName = "EconomicClass"
    print("Processing feature: ", featureName)
    preprocessedDataset[featureName] = currentDataset.apply(economicStatus, axis=1)
    # Reset indices:
    preprocessedDataset = preprocessedDataset.reset_index(drop=True)

    # Scale feature:
    if normalizeCategorical:
        standardizeFeature(featureName, preprocessedDataset, datasetName, categoricalScaler, encodersDictionary)

    # Process "Age" Feature:
    #   "NA" -> [-1, 0)
    #   Infant -> [0, 5)
    #   Child -> [5, 12)
    #   Teenager -> [12, 18)
    #   Young Adult -> [18, 35)
    #   Adult -> [35, 60)
    #   Senior -> [60, 100)

    ageBins = [-1, 0, 5, 12, 18, 35, 60, 100]
    binLabels = ["NA-Age", "Infant", "Child", "Teenager", "Young Adult", "Adult", "Senior"]

    # Replace "NaN" with "-0.5":
    currentDataset["Age"] = replaceFeatureValue(currentDataset[["Age"]], np.NaN, -0.5)
    # Segment feature into bins:
    labeledAge = pd.cut(currentDataset["Age"], bins=ageBins, labels=binLabels)
    # Convert series to data frame:
    currentDataset["Age-Labeled"] = pd.DataFrame(labeledAge)

    # Impute missing ages:
    featureName = "Age"
    imputerConfig = {"missingValue": -0.5, "strategy": "median"}
    imputeFeature(featureName, currentDataset, datasetName, imputerConfig, encodersDictionary)

    # Age groups feature:
    #   Group 1 -> [0, 18) Likely to be transported
    #   Group 2 -> [18, 26) Likely to be not transported
    #   Group 3 -> [26, 100) Could either be transported or not

    # The missing imputed median seams to be 27, careful to where bin
    # this value:

    ageBins = [-0.5, 18, 26, 100]
    binLabels = ["AgeGroup1-True", "AgeGroup2-False", "AgeGroup3-Either"]
    # Segment feature into bins:
    ageGroups = pd.cut(currentDataset["Age"], bins=ageBins, labels=binLabels)
    # Convert series to data frame:
    currentDataset["AgeGroups"] = pd.DataFrame(ageGroups)

    # Directly One-hot encode categorical features:
    print("One-hot encoding features...")

    categoricalFeatures = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Age-Labeled", "AgeGroups"]

    for currentFeature in categoricalFeatures:
        print("One-hot encoding feature:", currentFeature)
        # OHE feature:
        featureCategories = oheFeature(currentFeature, currentDataset, preprocessedDataset, datasetName,
                                       encodersDictionary)
        # Standardize feature?
        if normalizeBinary:
            standardizeFeature(featureCategories, preprocessedDataset, datasetName, binaryScaler, encodersDictionary)

    # Create transported likelihood features:
    featureName = "TransportedLikelihood"
    print("Processing feature: ", featureName)

    # Create the 2 new columns: "HighProba" and "LowProba"
    preprocessedDataset[["HighProba", "LowProba"]] = [0, 0]
    # Apply the rules for both columns
    preprocessedDataset[["HighProba", "LowProba"]] = preprocessedDataset.apply(getTransportProbas, axis=1,
                                                                               result_type='expand')

    # Add a new feature: TransportedLikelihood
    outputCategories = True  # Whether the output is plain differences or categories (-1, 0, 1):
    preprocessedDataset["TraHood"] = preprocessedDataset[["HighProba", "LowProba"]].apply(
        transportedLikelihood, args=(outputCategories,), axis=1)

    # Scaling of transported likelihood features:
    if normalizeCategorical:
        # "HighProba", "LowProba" are categories:
        standardizeFeature(["HighProba", "LowProba"], preprocessedDataset, datasetName, categoricalScaler,
                           encodersDictionary)
        # "TraHood" is categories:
        standardizeFeature("TraHood", preprocessedDataset, datasetName, categoricalScaler, encodersDictionary)

    # Process "Cabin" Feature:
    # Split cabin feature into 3 sub-features:
    cabinSplit = currentDataset["Cabin"].str.split("/", expand=True)

    # Get number of columns (should be 3 features):
    subFeatures = cabinSplit.shape[1]

    # Dataframe shallow copy:
    tempDataframe = cabinSplit

    # Name of sub-features:
    cabinFeatures = ["CabinDeck", "CabinNum", "CabinSide"]

    # Change temp dataframe columns to new feature names:
    tempDataframe.rename(
        columns={0: cabinFeatures[0], 1: cabinFeatures[1], 2: cabinFeatures[2]},
        inplace=True,
    )

    # The type of scaler for the "cabinNum" feature, which is a category...
    cabinScaler = "Standard"

    # Decides if NAs should be imputed or should be handled in a new column (NAs)
    imputeMissing = False

    # For the three sub-features created before:
    for i in range(subFeatures):

        # Set new feature name:
        featureString = cabinFeatures[i]
        print("Processing feature:", featureString)

        if i != 1:

            # NAs handling:
            if not imputeMissing:
                # Change NaNs for featureString + "-NA
                print("Extending NAs into a new column...")
                tempDataframe[featureString] = replaceFeatureValue(tempDataframe[featureString], np.NaN,
                                                                   featureString + "-NA")
            else:
                # Impute missing with most frequent value:
                print("Imputing NAs with most frequent value...")
                imputerConfig = {"missingValue": np.NaN, "strategy": "most_frequent"}
                imputeFeature(featureString, tempDataframe[[i]], datasetName, imputerConfig, encodersDictionary)

            # One-hot encoding:
            oheFeature(featureString, tempDataframe, tempDataframe, datasetName, encodersDictionary)

            # Standardize feature?
            if normalizeBinary:
                standardizeFeature(featureString, tempDataframe, datasetName, binaryScaler, encodersDictionary)

        else:

            # Extract the feature into one temp series (should be CabinNum):
            tempFeature = tempDataframe[featureString]

            # Series to DF:
            tempFeature = pd.DataFrame(tempFeature, columns=[featureString])

            # Impute missing values with median:
            imputerConfig = {"missingValue": np.NaN, "strategy": "median"}
            imputeFeature(featureString, tempFeature, datasetName, imputerConfig, encodersDictionary)

            # Scale feature... but "CabinNum" is a category...
            standardizeFeature(featureString, tempFeature, datasetName, cabinScaler, encodersDictionary)

            # Attach/append to outDataframe:
            tempDataframe[featureString + "-Std"] = tempFeature

    # Produce the final dataset slicing the temp dataset.
    # Slice from the new columns to the end, include all rows:
    tempDataframe = tempDataframe.iloc[:, 3:]

    # Reset indices:
    tempDataframe = tempDataframe.reset_index(drop=True)

    # Append/Concat to original dataframe based on left indices:
    preprocessedDataset = preprocessedDataset.join(tempDataframe)

    # Slice numerical features:
    numericalFeatures = currentDataset.loc[:, "RoomService":"VRDeck"].reset_index(drop=True)
    # Get feature names:
    featureNames = numericalFeatures.columns.values.tolist()

    # Impute missing values:
    # Maybe the strategy here could be median or most frequent, despite both values being 0:
    imputerConfig = {"missingValue": np.NaN, "strategy": "most_frequent"}
    imputeFeature(featureNames, numericalFeatures, datasetName, imputerConfig, encodersDictionary)

    # Should these features be included in the final dataset?
    if includeNumerical:
        # Do not change the original DF:
        tempDataframe = numericalFeatures.copy()
        # Before appending the features, scale them:
        standardizeFeature(featureNames, tempDataframe, datasetName, numericalScaler, encodersDictionary)
        # Append to final dataset:
        preprocessedDataset = preprocessedDataset.join(tempDataframe)

        # I'm free:
        del tempDataframe

    # Compute "Total Spending" feature:
    featureName = "TotalSpending"
    print("Processing feature: ", featureName)

    # Sum all numeric values:
    totalSpending = numericalFeatures.sum(axis=1)

    # Array to DataFrame, drop indices:
    totalSpending = pd.DataFrame(totalSpending, columns=[featureName]).reset_index(drop=True)

    # Scale feature:
    standardizeFeature(featureName, totalSpending, datasetName, numericalScaler, encodersDictionary)

    # Clean outliers from numerical features:
    threshList = ["lowerThresh", "upperThresh"]

    # Convert array to data frame, drop indices
    numericalFeatures = pd.DataFrame(data=numericalFeatures, columns=featureNames).reset_index(drop=True)

    # Cleaned data (w/o outliers):
    cleanedOutliers = pd.DataFrame()

    # Process the numerical features:
    for currentFeature in featureNames:

        if datasetName == "train":
            print(">> Fitting + Transforming: ", currentFeature)
            # Get Inter-quartile Range between Q1 and Q3:
            Q1 = numericalFeatures[currentFeature].quantile(0.26)
            Q3 = numericalFeatures[currentFeature].quantile(0.76)

            # Get IQR:
            IQR = Q3 - Q1
            # Compute lower and higher thresholds:
            whiskerWidth = 1.5
            lowerWhisker = Q1 - (whiskerWidth * IQR)
            upperWhisker = Q3 + (whiskerWidth * IQR)

            # Store thresholds:
            encodersDictionary[currentFeature] = {}
            encodersDictionary[currentFeature][threshList[0]] = lowerWhisker
            encodersDictionary[currentFeature][threshList[1]] = upperWhisker

        else:
            print(">> Fitting: ", currentFeature)
            # Get lower & upper IQR thresholds:
            lowerWhisker = encodersDictionary[currentFeature][threshList[0]]
            upperWhisker = encodersDictionary[currentFeature][threshList[1]]

        print(currentFeature, "->", "lowerThresh:", encodersDictionary[currentFeature][threshList[0]],
              "upperThresh", encodersDictionary[currentFeature][threshList[1]])

        # Replace all the values that are below the 25th percentile and above the 75th percentile of
        # the current feature:
        cleanedOutliers[currentFeature] = np.where(numericalFeatures[currentFeature] > upperWhisker, upperWhisker,
                                                   np.where(numericalFeatures[currentFeature] < lowerWhisker,
                                                            lowerWhisker, numericalFeatures[currentFeature]))

        # Set number of bins to segment the numerical features:
        binLabels = list(range(numericalBins))

        # Segment feature into len(binLabels) bins:
        totalBins = len(binLabels)
        encodedFeature = pd.cut(cleanedOutliers[currentFeature], bins=totalBins, labels=binLabels)

        # Append new feature:
        numericalFeatures[currentFeature + "-Binned"] = encodedFeature

    # Get the binned features only:
    tempDataframe = numericalFeatures.loc[:, "RoomService-Binned":"VRDeck-Binned"]

    # Scale features, which have been binned and are now categories:
    featureString = "Numerical Features"
    if normalizeCategorical:
        # Get feature names:
        featureNames = tempDataframe.columns.values.tolist()
        # Standardize:
        standardizeFeature(featureNames, tempDataframe, datasetName, categoricalScaler, encodersDictionary)

    # Attach total spending
    tempDataframe["TotalSpending"] = totalSpending

    # Append/Concat to original dataframe based on left indices:
    preprocessedDataset = preprocessedDataset.join(tempDataframe)

    # Filter least important features:
    if filterFeatures:
        print("Performing feature filtering for: ", datasetName)
        featureCounter = 0
        for featureName in featuresList:
            # Drop the feature:
            preprocessedDataset = preprocessedDataset.drop(featureName, axis=1)
            featureCounter += 1
        print("Dropped Features: ", featureCounter)

    # Get a quick description of the data:
    print(preprocessedDataset.info())
    # Shape:
    print("Dataset:", datasetName, " Samples: ", preprocessedDataset.shape[0], " Features: ",
          preprocessedDataset.shape[1])

    # Whole dataset into dictionary of datasets:
    datasets[datasetName]["dataset"] = preprocessedDataset

    print("Finished preprocessing for dataset: ", datasetName)

# Check out correlations:
corrDataFrame = datasets["train"]["dataset"]
corrDataFrame = corrDataFrame.join(datasets["train"]["labels"])

corrMatrix = corrDataFrame.corr()

corrList = corrMatrix[predictionLabel].sort_values(ascending=False)

print(corrList)
# Show total features, subtract the target label:
print("Total features: ", len(corrList) - 1)

if fitShallow:

    # Fit the model:
    print("Fitting model: ", modelType)

    # Dictionary of grid parameters:
    logisticParameters = {"solver": ["liblinear"], "C": np.logspace(-2, 2, 10), "penalty": ["l2", "l1"]}

    svmParameters = {"kernel": ["rbf"], "C": np.logspace(-0.2, 0.1, 10)[1:11],
                     }

    modelDictionary = {"SVM":
                           {"Model": svm.SVC(C=1.0, kernel="rbf", random_state=rng),
                            # {"Model": svm.SVC(C=0.11, kernel="rbf", gamma=0.06, random_state=rng),
                            # "Model": svm.SVC(C=1.7995852, kernel="rbf", random_state=rng),
                            "Params": svmParameters},
                       "LogisticRegression":
                           {"Model": LogisticRegression(solver="liblinear", penalty="l2", C=1.0,
                                                        random_state=rng),
                            # "Model": LogisticRegression(solver="liblinear", penalty="l1", C=0.05583914701751073,
                            #                            random_state=rng),
                            "Params": logisticParameters},
                       "SGD":  # SGDClassifier(loss="hinge", penalty="elasticnet", alpha=0.00015)
                           {"Model": SGDClassifier(loss="hinge", penalty="elasticnet", random_state=rng),
                            "Params": []},
                       "DecisionTree":
                           {"Model": DecisionTreeClassifier(criterion="gini", random_state=rng,
                                                            max_depth=10,
                                                            max_features=40,
                                                            # max_leaf_nodes=20,
                                                            min_samples_split=2,
                                                            min_samples_leaf=1,
                                                            splitter="best"
                                                            ),
                            "Params": []},
                       "RandomForest":
                           {"Model": RandomForestClassifier(random_state=rng,
                                                            n_estimators=48,
                                                            max_depth=20,
                                                            max_features=44,
                                                            # max_leaf_nodes=20,
                                                            min_samples_split=15,
                                                            ),
                            "Params": []},
                       "GradientBoost":
                           {"Model": GradientBoostingClassifier(random_state=rng,
                                                                n_estimators=150,
                                                                max_depth=4,
                                                                max_features=40,
                                                                # max_leaf_nodes=20,
                                                                learning_rate=0.1,
                                                                n_iter_no_change=15,
                                                                subsample=0.9
                                                                ),
                            "Params": []},

                       "AdaBoost":
                           {"Model": AdaBoostClassifier(random_state=rng,
                                                        base_estimator=DecisionTreeClassifier(max_depth=1),
                                                        n_estimators=100,
                                                        learning_rate=1.0
                                                        ),
                            "Params": []}
                       }

    # Classifiers that compose the ensemble:

    # classifierNames = ["SVM", "LogisticRegression", "RandomForest"]
    # classifierNames = ["SVM", "LogisticRegression", "DecisionTree"]

    # classifierNames = ["SVM", "DecisionTree"]

    classifierNames = ["GradientBoost", "RandomForest", "SVM"]

    # classifierNames = ["SVM", "LogisticRegression"]

    # Prepare the voting classifier:
    if modelType == "VotingClassifier":
        # Set the classifier list:
        classifierList = []
        # Prepare the classifier list:
        for classifierName in classifierNames:
            # Name + model, into the list:
            classifierTuple = (classifierName, modelDictionary[classifierName]["Model"])
            classifierList.append(classifierTuple)

        # Create the voting classifier
        votingClassifier = VotingClassifier(estimators=classifierList, voting=svmVoting)
        if svmVoting == "soft":
            if "SVM" in classifierNames:
                votingClassifier.named_estimators["SVM"].probability = True

        # Into the model dict:
        tempDict = {"Model": votingClassifier, "Params": []}
        modelDictionary["VotingClassifier"] = tempDict

    # Create the classifier model:
    currentModel = modelDictionary[modelType]["Model"]

    # Fit the model to the training data:
    trainLabels = datasets["train"]["labels"].values.ravel()  # From column to row
    trainFeatures = datasets["train"]["dataset"]

    if modelType == "SVM":
        currentModel.probability = True

    currentModel.fit(trainFeatures, trainLabels)

    # If Model is Random Forest, show feature importance:
    # If Model is Random Forest, show feature importance:
    if modelType == "RandomForest" and showFeatureImportance:
        showFeatures(currentModel, trainFeatures, featureScoreThreshold)

    # Check out the classifier's accuracy using cross-validation:
    print("[INFO] --- Cross-Validating Classifier...")
    modelAccuracy = cross_val_score(estimator=currentModel, X=trainFeatures, y=trainLabels, cv=cvFolds,
                                    n_jobs=parallelJobs, verbose=3)

    # Accuracy for each fold:
    print("[INFO] --- Fold Accuracy:")
    print(" ", modelAccuracy)

    print("[INFO] --- Mean & Std Dev Fold Accuracy:")
    cvMean = np.mean(np.array(modelAccuracy))
    cvStdDev = np.std(np.array(modelAccuracy))

    print(">> Mu: ", cvMean, "Sigma:", cvStdDev)

    # Stratified cross-validation:
    print("[INFO] --- Performing Stratified Cross-Validation...")
    runStratifiedCV = not (skipStratifiedCV) and not (computeThreshold)
    _, stratAccuracy, stratStdDev = stratifiedCrossValidation(currentModel, trainFeatures, trainLabels,
                                                              randomState=rng, splits=10, testSize=0.2,
                                                              runCV=runStratifiedCV,
                                                              verbose=False)

    # Store the best model so far:
    bestModel["Model"] = currentModel
    bestModel["Accuracy"] = cvMean

    if testSplit != -1:

        # Test current model:
        print("[INFO] --- Testing Classifier: ", modelType)
        print("Params: ", currentModel.get_params())

        if modelType == "GradientBoost":
            print("Trees used: ", currentModel.n_estimators_)

        testLabels = datasets["validation"]["labels"].values.ravel()  # From column to row
        testFeatures = datasets["validation"]["dataset"]

        if modelType == "VotingClassifier":
            print("Computing Voting Accuracies...")
            for classifierName, currentClassifier in modelDictionary[modelType]["Model"].named_estimators_.items():
                print(" ", classifierName, ":", currentClassifier.score(testFeatures, testLabels))

        modelScore = currentModel.score(testFeatures, testLabels)
        print(">> Validation Accuracy:", modelScore)

        # List of how many times the results must be run:
        runResults = [1]

        if computeThreshold:
            if modelType != "VotingClassifier":
                runResults.append(2)
            else:
                if svmVoting == "soft":
                    runResults.append(2)
                else:
                    print("svmVoting is set to hard. Skipping best threshold...")

        if dnnVote:

            # Load DNN model:
            modelFilename = "dnnModel.keras"
            modelPath = projectPath + modelFilename
            fileExists = os.path.exists(modelPath)

            if not fileExists:
                raise ValueError("Model file: " + modelPath + " does not exist.")
            else:
                print("Loading model from: " + modelPath)

            dnnModel = load_model(modelPath)

            # Attach results type to results list:
            runResults.append(3)

        # Best Validation Accuracy (so far):
        bestAccuracy = modelScore
        bestModel["Model"] = currentModel
        bestModel["Accuracy"] = bestAccuracy

        # Perform the result-gathering process:
        for r, resultsType in enumerate(runResults):

            print("Running results: ", r + 1)

            classificationThreshold = 0.0

            if resultsType == 1:
                print(">> Running Results Type: 1 (Default threshold)")
                # Get predictions:
                predictionProbabilities = currentModel.predict(testFeatures)

            elif resultsType == 2:
                print(">> Running Results Type: 2 (Best threshold)")
                # Get prediction probabilities:
                predictionProbabilities = currentModel.predict_proba(testFeatures)
                predictionProbabilities = predictionProbabilities[:, 1]
                # Compute the best classification threshold:
                classificationThreshold = bestThreshold(testLabels, predictionProbabilities)
                print("Best thresh: ", classificationThreshold)
                predictionProbabilities = (predictionProbabilities >= classificationThreshold).astype(int)

            elif resultsType == 3:

                print(">> Running Results Type: 3 (Shallow Classifiers + DNN)")

                # Get prediction probabilities:
                predictionProbabilities = currentModel.predict_proba(testFeatures)
                predictionProbabilities = predictionProbabilities[:, 1]
                # Row to column:
                predictionProbabilities = predictionProbabilities.reshape(-1, 1)

                # Feature data frame to tensors:
                tensorFeatures = tf.convert_to_tensor(testFeatures)
                # Dnn predictions:
                dnnPredictions = dnnModel.predict(tensorFeatures)

                # Average both predictions:
                predictionAverage = (dnnPredictions + predictionProbabilities) / 2.0
                # Set to output variable:
                predictionProbabilities = (predictionAverage >= 0.5).astype(int)
                predictionProbabilities = predictionProbabilities.flatten()

            # Common results displaying:

            print("[INFO] --- Computing Per sample accuracy...")

            # Get per sample accuracy:
            realClasses = perSampleAccuracy(predictionProbabilities, testLabels, False)
            print("Real Class labels counters: ")
            print(realClasses)

            # Get and display results:
            resultsDict = displayResults(predictionProbabilities, testLabels, modelScore, cvMean, cvStdDev, 10)

            # Get accuracy post-probabilities test (if performed)
            currentAccuracy = resultsDict["cmAccuracy"]
            if currentAccuracy > bestAccuracy:
                # Store model & accuracy:
                bestModel["Model"] = currentModel
                bestModel["Accuracy"] = currentAccuracy
                bestModel["Threshold"] = classificationThreshold
                print("Got better model: ", bestModel)

            # Display Stratified CV results:
            print("Strat. Mean Accuracy:", "{:.4f}".format(stratAccuracy), "Strat. Std.Dev:",
                  "{:.4f}".format(stratStdDev))

            # Plot confusion matrix:
            plotConfusionMatrix(testLabels, predictionProbabilities, currentModel)

        # Check parameters for grid search:
        gridParameters = modelDictionary[modelType]["Params"]

        # Grid search:
        if runGridSearch and gridParameters:
            # Hyperparameter optimization using Grid Search:
            print("[INFO] --- Running Grid Search for: ", modelType)

            optimizedModels = {"SVM": svm.SVC(random_state=rng),
                               "LogisticRegression": LogisticRegression(random_state=rng, max_iter=400)}

            optimizedModel = optimizedModels[modelType]

            optimizedModel = RandomizedSearchCV(optimizedModel, gridParameters, cv=3,
                                                n_jobs=parallelJobs)
            optimizedModel.fit(trainFeatures, trainLabels)

            # Print hyperparameters & accuracy:
            print("[INFO] --- Grid Search Best Parameters:")
            print("", optimizedModel.best_params_)

            # Check out the reggressor accuracy using cross-validation:
            print("[INFO] --- [Post-Grid Search] Cross-Validating Classifier...")
            modelAccuracy = cross_val_score(estimator=optimizedModel, X=trainFeatures, y=trainLabels,
                                            cv=cvFolds,
                                            n_jobs=parallelJobs, verbose=3)

            # Accuracy for each fold:
            print("[INFO] --- [Post-Grid Search] Fold Accuracy:")
            print(" ", modelAccuracy)

            print("[INFO] --- [Post-Grid Search] Mean & Std Dev Fold Accuracy:")
            cvMean = np.mean(np.array(modelAccuracy))
            cvStdDev = np.std(np.array(modelAccuracy))

            print(">> Mu: ", cvMean, "Sigma:", cvStdDev)

            # Get score:
            modelScore = optimizedModel.score(testFeatures, testLabels)

            # Get predictions:
            modelPredictions = optimizedModel.predict(testFeatures)

            # Get per sample accuracy:
            print("[INFO] --- Computing Per sample accuracy...")
            # Get per sample accuracy:
            realClasses = perSampleAccuracy(modelPredictions, testLabels, False)
            print("Real Class labels counters: ")
            print(realClasses)

            # Get and display results:
            resultsDict = displayResults(modelPredictions, testLabels, modelScore, cvMean, cvStdDev, 3)
            currentAccuracy = resultsDict["cmAccuracy"]

            if currentAccuracy > bestAccuracy:
                # Store model & accuracy:
                bestModel["Model"] = optimizedModel
                bestModel["Accuracy"] = currentAccuracy
                bestModel["Threshold"] = 0.0
                print("Got better model: ", optimizedModel.__class__.__name__)

            # Plot confusion matrix:
            plotConfusionMatrix(testLabels, modelPredictions, optimizedModel)

if runDNN:
    print("Running DNN...")

    # Get train, validation datasets & labels:
    trainFeatures = datasets["train"]["dataset"]
    trainLabels = datasets["train"]["labels"]

    valFeatures = datasets["validation"]["dataset"]
    valLabels = datasets["validation"]["labels"]

    # Feature data frames to tensors:
    trainFeatures = tf.convert_to_tensor(trainFeatures)
    valFeatures = tf.convert_to_tensor(valFeatures)

    # Set the dnn architecture:
    model = keras.Sequential(
        [
            layers.Dense(units=64, kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(0.5),

            layers.Dense(units=32, kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(0.5),

            # layers.Dense(units=16, kernel_initializer="he_normal"),
            # layers.BatchNormalization(),
            # layers.ReLU(),
            # layers.Dropout(0.5),

            layers.Dense(1, activation="sigmoid")

        ]
    )

    # Compile the model:
    optimizer = Adam(learning_rate=dnnLearningRate)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    # Model Checkpoint,
    # Model weights are saved at the end of every epoch, if it's the best seen so far:
    checkpointFilename = "bestSoFar.keras"
    modelCheckpoint = ModelCheckpoint(filepath=projectPath + checkpointFilename,
                                      save_weights_only=False,
                                      monitor="val_accuracy",
                                      mode="max", save_best_only=True,
                                      verbose=0)

    # Fit the model:
    history = model.fit(trainFeatures,
                        trainLabels,
                        epochs=totalEpochs,
                        batch_size=dnnBatchSize,
                        validation_data=(valFeatures, valLabels),
                        callbacks=[modelCheckpoint])

    loss, accuracy = model.evaluate(valFeatures, valLabels)
    print("Test loss:", loss)
    print("Test accuracy:", accuracy)

    # Plot the learning curves:
    # Plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()

    # Get the historical data:
    N = np.arange(0, totalEpochs)

    # Plot values:
    plt.plot(N, history.history["loss"], label="train_loss")
    plt.plot(N, history.history["val_loss"], label="val_loss")
    plt.plot(N, history.history["accuracy"], label="train_acc")
    plt.plot(N, history.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")

    # Save plot to disk:
    plotPath = projectPath + "lossGraph.png"
    print("Saving model loss plot to:" + plotPath)
    plt.savefig(plotPath)
    plt.show()

if getFinalPredictions:

    print(">> Computing final predictions...")
    print(">> Best model was: ", bestModel["Model"].__class__.__name__, "(" + str(bestModel["Accuracy"]) + ")")

    # Fit the model to the test data:
    testFeatures = datasets["test"]["dataset"]

    # Check threshold:
    finalModel = bestModel["Model"]
    binaryThreshold = bestModel["Threshold"]

    # Predict using default threshold or best threshold:
    epsilon = 0.001
    thresholdDifference = abs(binaryThreshold - 0.0)
    if thresholdDifference <= epsilon:
        print("Predicting final values with default binary threshold.")
        finalPredictions = finalModel.predict(testFeatures)
    else:
        print("Predicting final values with best binary threshold: ", binaryThreshold)
        finalPredictions = (finalModel.predict_proba(testFeatures)[:, 1] >= binaryThreshold).astype(int)

    # Convert numpy array to dataframe:
    finalPredictionsDataFrame = pd.DataFrame(finalPredictions)

    # Transpose original array (row to column):
    finalPredictions = finalPredictions.reshape(-1, 1)

    # Change 1 -> True, 0 -> False:
    finalPredictionsDataFrame = replaceFeatureValue(finalPredictionsDataFrame, 1, True)
    finalPredictionsDataFrame = replaceFeatureValue(finalPredictionsDataFrame, 0, False)

    # Attach predictions:
    passengerList[predictionLabel] = finalPredictionsDataFrame

    # Write CSV:
    if writeOutfile:
        # Get classifier name:
        classifierName = bestModel["Model"].__class__.__name__
        # Get date NOW:
        dateNow = time.strftime("%Y-%m-%d-%H%M")

        outFilename = projectPath + outFilename + "-" + classifierName + "_" + dateNow + ".csv"
        print(">> Writing output file: ", outFilename)
        passengerList.to_csv(outFilename, index=False)

print(">> Done. Fuck you.")

# File        :   spaceshipTitanic.py
# Version     :   1.2.0
# Description :   Solution for Kaggle"s Spaceship Titanic problem
#                 (https://www.kaggle.com/competitions/spaceship-titanic)

# Date:       :   Nov 16, 2023
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT

import numpy as np
import pandas as pd
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop
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
from sklearn.preprocessing import StandardScaler, RobustScaler
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


def economic_status(person):
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


# Project Path:
projectPath = "D://dataSets//spaceTitanic//"
# Output Path:
outFilename = "outPredictions"
# Write final CSV?
writeOutfile = True

# File Names:
datasetNames = ["train", "validation", "test"]

# Dataset file extension:
fileExtension = ".csv"

# Prediction label:
predictionLabel = "Transported"

# Script options
randomSeed = 42
rng = np.random.RandomState(randomSeed)

# Test split:
# -1 uses all train dataset for training (no validation):
testSplit = 0.2

# Model type:
modelType = "VotingClassifier"
svmVoting = "soft"

getFinalPredictions = False
fitShallow = True

# Use dnn to predict a new feature?
dnnFeature = False

# Should the DNN be run?
runDNN = False
# DNN epochs:
totalEpochs = 100
# Batch size:
dnnBatchSize = 64
# Learning rate:
dnnLearningRate = 0.0002

# Store best model here:
bestModel = {"Model": None, "Accuracy": 0.0, "Threshold": 0.0}

# Compute the best threshold?
computeThreshold = False

numericalBins = 5

# Skip stratified CV:
skipStratifiedCV = True

# Cross-validation folds:
cvFolds = 10
runGridSearch = True
# Cross-validation parallel jobs (1 per core):
parallelJobs = 5

# Feature Selection:
showFeatureImportance = True
featureScoreThreshold = 0.98
# featuresList = ["C", "Infant", "Child", "NA-Age", "55 Cancri e", "NA-CryoSleep", "B", "TRAPPIST-1e", "Young Adult",
#                 "Adult", "D", "PSO J318.5-22", "VIP-FALSE", "Senior", "NA-Cabin-2", "NA-VIP", "NA-HomePlanet",
#                 "NA-Destination", "NA-Cabin-0", "VIP-TRUE", "Teenager", "A", "T"]

# The list of features to drop:
featuresList = ["T"]
filterFeatures = False

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

# Load DNN model:
modelFilename = "dnnModel.keras"
modelPath = projectPath + modelFilename
fileExists = os.path.exists(modelPath)

if not fileExists:
    raise ValueError("Model file: " + modelPath + " does not exist.")
else:
    print("Loading model from: " + modelPath)

dnnModel = load_model(modelPath)

# Store here the test passenger columns for final CSV:
passengerList = pd.DataFrame()

for d, datasetName in enumerate(datasetNames):

    print("Processing Dataset:", datasetName)

    # Select dataset type:
    if datasetName == "train":

        # Read the cvs file:
        currentDataset = pd.read_csv(projectPath + datasetName + fileExtension)

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
    preprocessedDataset[featureName] = currentDataset.apply(economic_status, axis=1)
    # Reset indices:
    preprocessedDataset = preprocessedDataset.reset_index(drop=True)

    # Scale feature:
    if datasetName == "train":
        print(">> Fitting + Transforming: ", featureName)
        # Set scaler:
        currentScaler = StandardScaler()
        preprocessedDataset[featureName] = currentScaler.fit_transform(preprocessedDataset[[featureName]])
        encodersDictionary[featureName + "-Scaler"] = currentScaler
        # print(currentScaler.mean_)
    else:
        print(">> Transforming: ", featureName)
        # Get Scaler:
        currentScaler = encodersDictionary[featureName + "-Scaler"]
        preprocessedDataset[featureName] = currentScaler.transform(preprocessedDataset[[featureName]])

    # Process "Age" Feature:
    # Segment "age" feature into the following bins:
    #   "NA" -> [-1, 0)
    #   Infant -> [0, 5)
    #   Child -> [5, 12)
    #   Teenager -> [12, 18)
    #   Young Adult -> [18, 35)
    #   Adult -> [35, 60)
    #   Senior -> [60, 100)
    ageBins = [-1, 0, 5, 12, 18, 35, 60, 100]
    binLabels = ["NA-Age", "Infant", "Child", "Teenager", "Young Adult", "Adult", "Senior"]
    currentDataset["Age-Labeled"] = processAge(currentDataset, ageBins, binLabels)

    # Directly One-hot encode categorical features:
    print("One-hot encoding features...")
    categoricalFeatures = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Age-Labeled"]

    for currentFeature in categoricalFeatures:
        print("One-hot encoding feature:", currentFeature)

        if datasetName == "train":
            print(">> Fitting + Transforming: ", currentFeature)
            # Create the encoder object:
            currentEncoder = OneHotEncoder()
            # Fit + transform to feature:
            encodedFeature = currentEncoder.fit_transform(currentDataset[[currentFeature]])

            # Store encoder into dictionary:
            if currentFeature not in encodersDictionary:
                encodersDictionary[currentFeature] = currentEncoder

        else:
            print(">> Fitting: ", currentFeature)
            # Create the encoder object:
            currentEncoder = encodersDictionary[currentFeature]
            # Transform the feature:
            encodedFeature = currentEncoder.transform(currentDataset[[currentFeature]])

        # Convert to array:
        encodedFeature = encodedFeature.toarray()

        # Print categories:
        print(currentFeature, "Categories: ", currentEncoder.categories_)

        # Append to categorical dataset:
        preprocessedDataset[currentEncoder.categories_[0]] = encodedFeature

    # Process "Cabin" Feature:
    # Split cabin feature into 3 sub-features:
    cabinSplit = currentDataset["Cabin"].str.split("/", expand=True)

    # Dataframe shallow copy:
    tempDataframe = cabinSplit

    # Transformers list:
    cabinTransformers = ["-Imputer", "-Scaler"]

    # For the three sub-features created before:
    for i in range(3):

        # Set new feature name:
        featureString = "NA-Cabin-" + str(i)
        print("Processing feature:", featureString)

        if i != 1:

            # Change NaNs for "NA-Cabin" + str(i)
            tempDataframe[[i]] = replaceFeatureValue(tempDataframe[[i]], np.NaN, featureString)

            if datasetName == "train":
                print(">> Fitting + Transforming: ", featureString)
                # Create encoder object and apply it to the dataframe:
                currentEncoder = OneHotEncoder()
                currentEncoded = currentEncoder.fit_transform(tempDataframe[[i]])

                # Store encoder into dictionary:
                if featureString not in encodersDictionary:
                    encodersDictionary[featureString] = currentEncoder

            else:
                print(">> Fitting: ", featureString)
                # Get encoder:
                currentEncoder = encodersDictionary[featureString]
                # Transform feature:
                currentEncoded = currentEncoder.transform(tempDataframe[[i]])

            # Encoding to array:
            currentEncoded = currentEncoded.toarray()

            # Check out categories:
            print(featureString, "->", currentEncoder.categories_)

            # Attach/append to outDataframe:
            tempDataframe[currentEncoder.categories_[0]] = currentEncoded

        else:

            # Impute missing values with median:
            tempFeature = tempDataframe[[i]]

            if datasetName == "train":
                print(">> Fitting + Transforming: ", cabinTransformers[0])
                # Set imputer:
                currentImputer = SimpleImputer(missing_values=np.NaN, strategy="median")
                # Fit + transform feature:
                tempFeature = currentImputer.fit_transform(tempFeature)
                encodersDictionary[featureString + cabinTransformers[0]] = currentImputer

                # Scale feature:
                currentScaler = StandardScaler()
                tempFeature = currentScaler.fit_transform(tempFeature)
                encodersDictionary[featureString + cabinTransformers[1]] = currentScaler

                # # Store transformers into dictionary:
                # for n in cabinTransformers:
                #     dictKey = featureString + n
                #     if dictKey not in encodersDictionary:
                #         encodersDictionary[dictKey] = currentImputer
            else:
                print(">> Fitting: ", cabinTransformers[0])
                for n in cabinTransformers:
                    transformerName = featureString + n
                    print("Applying transformer: ", transformerName)
                    # Set transformer:
                    currentTransformer = encodersDictionary[transformerName]
                    # Transform feature:
                    tempFeature = currentTransformer.transform(tempFeature)

            # Attach/append to outDataframe:
            tempDataframe["CabinNum"] = tempFeature

    # Produce the final dataset slicing the temp dataset.
    # Slice from the new columns to the ends, include all rows:
    tempDataframe = tempDataframe.iloc[:, 3:]

    # Reset indices:
    tempDataframe = tempDataframe.reset_index(drop=True)

    # Append/Concat to original dataframe based on left indices:
    # trainDatasetCopy = pd.merge(trainDatasetCopy, cabinSplit, left_index=True, right_index=True)
    preprocessedDataset = preprocessedDataset.join(tempDataframe)

    # Numerical features:
    featureString = "Numerical Features"
    threshList = ["lowerThresh", "upperThresh"]

    # Slice numerical features:
    numericalFeatures = currentDataset.loc[:, "RoomService":"VRDeck"]
    # Get feature names:
    featureNames = numericalFeatures.columns.values.tolist()

    # Total Spending:
    featureName = "totalSpending"

    # Impute missing values:
    if datasetName == "train":

        print(">> Fitting + Transforming: ", featureString)
        # Set imputer,
        # Maybe the strategy here could be median or most frequent, despite both values being 0:
        currentImputer = SimpleImputer(missing_values=np.NaN, strategy="most_frequent")
        # Fit + transform transformer:
        numericalFeatures = currentImputer.fit_transform(numericalFeatures)

        # Store imputer into dictionary:
        if featureString not in encodersDictionary:
            encodersDictionary[featureString + "-Imputer"] = currentImputer

        print("Processing feature: ", featureName)
        # Sum all numeric values:
        totalSpending = numericalFeatures.sum(axis=1)
        # Array to DataFrame:
        totalSpending = pd.DataFrame(totalSpending)

        # Scale feature:
        print(">> Fitting + Transforming: ", featureName)

        # Set Scaler:
        currentScaler = StandardScaler()
        totalSpending = currentScaler.fit_transform(totalSpending)

        # Store scaler into dictionary:
        encodersDictionary[featureName + "-Scaler"] = currentScaler

    else:

        print(">> Fitting: ", featureString)
        # Set imputer:
        currentImputer = encodersDictionary[featureString + "-Imputer"]
        # Fit + transform transformer:
        numericalFeatures = currentImputer.transform(numericalFeatures)

        # Total Spending:
        print("Processing feature: ", featureName)
        # Sum all numeric values:
        totalSpending = numericalFeatures.sum(axis=1)
        # Array to DataFrame:
        totalSpending = pd.DataFrame(totalSpending)

        # Scale feature:
        print(">> Transforming: ", featureName)

        # Get Scaler:
        currentScaler = encodersDictionary[featureName + "-Scaler"]
        totalSpending = currentScaler.transform(totalSpending)

    # Convert array to data frame:
    numericalFeatures = pd.DataFrame(data=numericalFeatures, columns=featureNames)
    # Set number of bins to segment the numerical fatures:
    binLabels = list(range(numericalBins))

    # Cleaned data (w/o outliers):
    cleanedOutliers = pd.DataFrame()

    # Append/Concat to original dataframe based on left indices:
    # preprocessedDataset = preprocessedDataset.join(numericalFeatures)

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

        # Segment feature into len(binLabels) bins:
        totalBins = len(binLabels)
        encodedFeature = pd.cut(cleanedOutliers[currentFeature], bins=totalBins, labels=binLabels)

        # Append new feature:
        numericalFeatures[currentFeature + "-Bined"] = encodedFeature

    # Get the binned features only:
    tempDataframe = numericalFeatures.loc[:, "RoomService-Bined":"VRDeck-Bined"]

    # Scale features:
    if datasetName == "train":
        print(">> Fitting + Transforming: ", featureString)
        # Set scaler:
        currentScaler = StandardScaler()
        # Fit + transform:
        tempDataframe = currentScaler.fit_transform(tempDataframe)

        # Store scaler into dictionary:
        if featureString not in encodersDictionary:
            encodersDictionary[featureString + "-Scaler"] = currentScaler

    else:
        print(">> Fitting: ", featureString)
        # Set scaler:
        currentScaler = encodersDictionary[featureString + "-Scaler"]
        # Transform:
        tempDataframe = currentScaler.transform(tempDataframe)

    # Prepare the final dataframe (processed data + column names):
    tempDataframe = pd.DataFrame(data=tempDataframe, columns=featureNames)

    # Attach total spending
    tempDataframe["TotalSpending"] = totalSpending

    # # Bin total spending:
    # spendingBins = 7
    # spendingLabels = list(range(spendingBins))
    # tempDataframe["TotalSpending"] = pd.cut(tempDataframe["TotalSpending"], bins=spendingBins, labels=spendingLabels)

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

    # Pass dataset through DNN:
    if dnnFeature:
        print("Dataset: ", datasetName, "Computing DNN predictions.")
        # Feature data frame to tensors:
        tensorFeatures = tf.convert_to_tensor(preprocessedDataset)

        # Dnn predictions:
        dnnPredictions = dnnModel.predict(tensorFeatures)

        # Add model predictions to dataset:
        preprocessedDataset["Dnn"] = dnnPredictions

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
    logisticParameters = {"solver": ["liblinear"], "C": np.logspace(-1, 0, 40)[1:16], "penalty": ["l2"]}
    svmParameters = {"kernel": ["linear", "rbf"],
                     "C": np.logspace(-1.0, 0.4, 10)}
    # svmParameters = {"kernel": ["rbf"], "C": [2.1544346900318843], "gamma": np.logspace(-2.1, 0.01, 20)[1:10]}

    modelDictionary = {"SVM":  # svm.SVC(C=0.8, kernel="rbf")
        {  # "Model": svm.SVC(C=2.1544, gamma=0.0614, kernel="rbf", random_state=rng),
            "Model": svm.SVC(C=1.7995852, kernel="rbf", random_state=rng),
            "Params": svmParameters},
        "LogisticRegression":
            {"Model": LogisticRegression(solver="liblinear", penalty="l2", C=0.10608183551394486, random_state=rng),
             # "Model": LogisticRegression(solver="liblinear", penalty="l1", C=0.05583914701751073,
             #                            random_state=rng),
             "Params": logisticParameters},
        "SGD":  # SGDClassifier(loss="hinge", penalty="elasticnet", alpha=0.00015)
            {"Model": SGDClassifier(loss="hinge", penalty="elasticnet", alpha=0.00008, random_state=rng),
             "Params": []},
        "DecisionTree":
            {"Model": DecisionTreeClassifier(criterion="gini", random_state=rng,
                                             max_depth=4,
                                             max_features=35,
                                             # max_leaf_nodes=10,
                                             # min_samples_split=2,
                                             # min_samples_leaf=2,
                                             # splitter="best"
                                             ),
             "Params": []},
        "RandomForest":
            {"Model": RandomForestClassifier(random_state=rng,
                                             n_estimators=60,
                                             max_depth=10,
                                             max_features=10,
                                             max_leaf_nodes=30,
                                             min_samples_split=30,
                                             ),
             "Params": []},
        "GradientBoost":
            {"Model": GradientBoostingClassifier(random_state=rng,
                                                 n_estimators=150,
                                                 max_depth=4,
                                                 max_features="auto",
                                                 max_leaf_nodes=20,
                                                 learning_rate=0.09,
                                                 n_iter_no_change=10,
                                                 subsample=1.0
                                                 ),
             "Params": []},

        "AdaBoost":
            {"Model": AdaBoostClassifier(random_state=rng,
                                         base_estimator=DecisionTreeClassifier(max_depth=2),
                                         n_estimators=100,
                                         learning_rate=0.5
                                         ),
             "Params": []}
    }

    # Classifiers that compose the ensemble:"
    # classifierNames = ["SVM", "LogisticRegression", "RandomForest"]
    # classifierNames = ["SVM", "LogisticRegression", "DecisionTree"]

    classifierNames = ["GradientBoost", "AdaBoost"]

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
        runResults = [False]

        if computeThreshold:
            if modelType != "VotingClassifier":
                runResults.append(True)
            else:
                if svmVoting == "soft":
                    runResults.append(True)
                else:
                    print("svmVoting is set to hard. Skipping best threshold...")

        # Best Validation Accuracy (so far):
        bestAccuracy = modelScore
        bestModel["Model"] = currentModel
        bestModel["Accuracy"] = bestAccuracy

        # Perform the result-gathering process:
        for r, computeProbas in enumerate(runResults):

            print("Running results: ", r + 1)

            classificationThreshold = 0.0

            if computeProbas:
                # Get prediction probabilities:
                predictionProbabilities = currentModel.predict_proba(testFeatures)
                predictionProbabilities = predictionProbabilities[:, 1]
                # Compute the best classification threshold:
                classificationThreshold = bestThreshold(testLabels, predictionProbabilities)
                print("Best thresh: ", classificationThreshold)
                predictionProbabilities = (predictionProbabilities >= classificationThreshold).astype(int)

            else:
                # Get predictions:
                predictionProbabilities = currentModel.predict(testFeatures)

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
                bestModel["Model"] = currentModel
                bestModel["Accuracy"] = currentAccuracy
                bestModel["Threshold"] = 0.0
                print("Got better model: ", bestModel.__class__.__name__)

            # Plot confusion matrix:
            plotConfusionMatrix(testLabels, modelPredictions, optimizedModel)

if runDNN:
    print("Running DNN...")

    # Get train, validation datasets & labels:
    trainFeatures = datasets["train"]["dataset"]
    trainLabels = datasets["train"]["labels"]

    testFeatures = datasets["validation"]["dataset"]
    testLabels = datasets["validation"]["labels"]

    # Feature data frames to tensors:
    trainFeatures = tf.convert_to_tensor(trainFeatures)
    testFeatures = tf.convert_to_tensor(testFeatures)

    # Set the dnn architecture:
    model = keras.Sequential(
        [
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(16, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(8, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(4, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid")

        ]
    )

    # Compile the model:

    optimizer = RMSprop(learning_rate=dnnLearningRate)
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
                        validation_data=(testFeatures, testLabels),
                        callbacks=[modelCheckpoint])

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

    # Pass dataset through DNN:
    if dnnFeature:
        print("Computing DNN predictions.")

        # Feature data frame to tensors:
        tensorFeatures = tf.convert_to_tensor(testFeatures)

        # Dnn predictions:
        dnnPredictions = dnnModel.predict(tensorFeatures)

        # Add model predictions to dataset:
        testFeatures["Dnn"] = dnnPredictions

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

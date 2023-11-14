# File        :   spaceshipTitanic.py
# Version     :   1.1.8
# Description :   Solution for Kaggle's Spaceship Titanic problem
#                 (https://www.kaggle.com/competitions/spaceship-titanic)

# Date:       :   Nov 13, 2023
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import IsolationForest

from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.model_selection import cross_val_score

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, RandomizedSearchCV, HalvingGridSearchCV, \
    HalvingRandomSearchCV

from sklearn.base import clone

from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_score, \
    recall_score, precision_recall_curve, f1_score

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


# Performs stratified cross-validation:
def stratifiedCrossValidation(currentModel, trainFeatures, trainLabels, randomState, splits=10, testSize=0.2,
                              verbose=False):
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
def plotConfusionMatrix(testLabels, model):
    modelPredictions = model.predict(testFeatures)
    confusionMatrix = confusion_matrix(testLabels, modelPredictions, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=model.classes_)
    disp.plot()
    plt.show()


# Get per sample accuracy and returns counter of test (real) labels:
def perSampleAccuracy(testFeatures, testLabels, currentModel, computeProbas=False):
    if computeProbas:
        # Get prediction probabilities:
        predictionProbabilities = currentModel.predict_proba(testFeatures)
    else:
        predictionProbabilities = currentModel.predict(testFeatures)

    # Real classes counter:
    classesCounter = {"0": 0, "1": 0}

    # List of accuracies per sample:
    sampleAccuracies = []

    # Print the predicted class, real class and probabilities per sample:
    for i in range(len(testFeatures)):

        # Get sample max probability:
        sampleProbability = np.max(predictionProbabilities[i])
        # Into the list:
        sampleAccuracies.append(sampleProbability)
        # Get predicted class:
        sampleClass = np.argmax(predictionProbabilities[i])
        # Get real class:
        realClass = testLabels[i]
        # Into class counter:
        classesCounter[str(realClass)] += 1
        # Print missmatch:
        missmatch = ""
        if realClass != sampleClass:
            missmatch = " <-"
        # Print the info:
        print(" Sample:", i, "Truth:", realClass, "Predicted:", sampleClass,
              "(Proba: " + "{:.4f}".format(sampleProbability) + ")" + missmatch)

    return classesCounter


# Computes and prints results:
def displayResults(currentModel, testFeatures, testLabels, realClasses, cvMean, cvStdDev, cvFolds):
    # Print the confusion matrix array:
    modelPredictions = currentModel.predict(testFeatures)
    cmArray = confusion_matrix(testLabels, modelPredictions)  # normalize='pred'
    print("Confusion Matrix: ")
    print(cmArray)

    # Get accuracy from CM:
    accuracy = (cmArray[0][0] + cmArray[1][1]) / len(testLabels)
    print("Class labels counters: ")
    print(realClasses)

    # Compute precision & recall:
    modelPrecision = precision_score(testLabels, modelPredictions)
    modelRecall = recall_score(testLabels, modelPredictions)

    # Compute F1 score:
    f1Score = f1_score(testLabels, modelPredictions)

    # Get model score:
    modelScore = currentModel.score(testFeatures, testLabels)

    # Print the results:
    dateNow = time.strftime("%Y-%m-%d %H:%M")
    print("---------------------------------------------------------- ")
    print("Results Test time: " + dateNow)
    print("Precision: ", modelPrecision)
    print("Recall: ", modelRecall)
    print("F1: ", f1Score)
    print("---------------------------------------------------------- ")
    print("Validation CM Accuracy:", accuracy)
    print("Validation Accuracy:", modelScore)

    print(">> Cross-validation Mean (" + str(cvFolds) + " Folds): ", "{:.4f}".format(cvMean),
          "StdDev: ", "{:.4f}".format(cvStdDev))


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
    plt.barh(lowRange, lowScores, color="red", edgecolor='red')
    plt.barh(highRange, highScores, color="green", edgecolor='green')

    # Add feature names to the vertical axis:
    plt.yticks(range(len(featuresSorted)), featuresSorted)
    plt.show()


# Project Path:
projectPath = "D://dataSets//spaceTitanic//"
# Output Path:
outFilename = projectPath + "outPredictions5.csv"
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
rng = np.random.RandomState(randomSeed)

# Test split:
# -1 uses all train dataset for training (no validation):
testSplit = 0.2

# Model type:
modelType = "VotingClassifier"
svmVoting = "hard"

numericalBins = 5
# runGridSearch = False

# Feature Selection:
showFeatureImportance = True
featureScoreThreshold = 0.98
# featuresList = ['C', 'Infant', 'Child', 'NA-Age', '55 Cancri e', 'NA-CryoSleep', 'B', 'TRAPPIST-1e', 'Young Adult',
#                 'Adult', 'D', 'PSO J318.5-22', 'VIP-FALSE', 'Senior', 'NA-Cabin-2', 'NA-VIP', 'NA-HomePlanet',
#                 'NA-Destination', 'NA-Cabin-0', 'VIP-TRUE', 'Teenager', 'A', 'T']

# The list of features to drop:
featuresList = ['NA-Cabin-0']
filterFeatures = False

# Cross-validation folds:
cvFolds = 10

runGridSearch = False
# Cross-validation parallel jobs (1 per core):
parallelJobs = 5

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

        # if currentFeature == "CryoSleep":
        #
        #     keyName = "CryoImputer-" + currentFeature
        #
        #     if datasetName == "train":
        #         currentImputer = SimpleImputer(missing_values=replacementString, strategy="most_frequent")
        #         currentDataset[currentFeature] = currentImputer.fit_transform(
        #             currentDataset[currentFeature].values.reshape(-1, 1))[:, 0]
        #         # Store encoder into dictionary:
        #         if keyName not in encodersDictionary:
        #             encodersDictionary[keyName] = currentImputer
        #     else:
        #         # Create the encoder object:
        #         currentImputer = encodersDictionary[keyName]
        #         # Transform the feature:
        #         currentDataset[currentFeature] = currentImputer.transform(
        #             currentDataset[currentFeature].values.reshape(-1, 1))[:, 0]

        # Additionally, replace "True" and "False" in "CryoSleep/VIP" Column:
        if currentFeature == "CryoSleep" or currentFeature == "VIP":
            print("Replacing True/False in: ", currentFeature)
            currentDataset[currentFeature] = replaceFeatureValue(currentDataset[currentFeature], True,
                                                                 currentFeature + "-TRUE")
            currentDataset[currentFeature] = replaceFeatureValue(currentDataset[currentFeature], False,
                                                                 currentFeature + "-FALSE")

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

    # Prepare the final preprocessed dataset:
    preprocessedDataset = pd.DataFrame()

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

    else:
        print(">> Fitting: ", featureString)
        # Set imputer:
        currentImputer = encodersDictionary[featureString + "-Imputer"]
        # Fit + transform transformer:
        numericalFeatures = currentImputer.transform(numericalFeatures)

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

    # Store in dataset dictionary:
    datasets[datasetName]["dataset"] = preprocessedDataset

    print("Finished preprocessing for dataset: ", datasetName)

# Fit the model:
print("Fitting model: ", modelType)

# Dictionary of grid parameters:
logisticParameters = {"solver": ["liblinear"], "C": np.logspace(-1.5, 1, 50)[1:10], "penalty": ["l1", "l2"]}
# svmParameters = {"kernel": ["rbf"], "C": np.logspace(-2, 10, 2), "gamma": np.logspace(-9, 3, 3)}
svmParameters = {"kernel": ["rbf"], "C": [2.1544346900318843], "gamma": np.logspace(-2.1, 0.01, 20)[1:10]}

modelDictionary = {"SVM":  # svm.SVC(C=0.8, kernel="rbf")
                       {"Model": svm.SVC(C=2.1544, gamma=0.0614, kernel="rbf", random_state=rng),
                        # "Model": svm.SVC(C=0.8, kernel="rbf"),
                        "Params": svmParameters},
                   "LogisticRegression":
                       {"Model": LogisticRegression(solver="liblinear", penalty="l2", C=1.0, random_state=rng),
                        # "Model": LogisticRegression(solver="liblinear", penalty="l1", C=0.05583914701751073,
                        #                            random_state=rng),
                        "Params": logisticParameters},
                   "SGD":  # SGDClassifier(loss="hinge", penalty="elasticnet", alpha=0.00015)
                       {"Model": SGDClassifier(loss="hinge", penalty="elasticnet", alpha=0.00015, random_state=rng),
                        "Params": []},
                   "DecisionTree":
                       {"Model": DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=rng,
                                                        max_features=30, max_leaf_nodes=25,
                                                        min_samples_split=2, min_samples_leaf=5,
                                                        splitter="best"),
                        "Params": []},
                   "RandomForest":
                       {"Model": RandomForestClassifier(n_estimators=20, max_depth=10, random_state=rng,
                                                        max_features=10, max_leaf_nodes=30,
                                                        min_samples_split=20,
                                                        ),
                        "Params": []}
                   }

# Classifiers that compose the ensemble:
classifierNames = ["SVM", "LogisticRegression", "DecisionTree"]

# Prepare the voting classifier:
if modelType == "VotingClassifier":
    # Set the classifier list:
    classifierList = []
    # Prepare the classifier list:
    for classifierNames in classifierNames:
        # Name + model, into the list:
        classifierTuple = (classifierNames, modelDictionary[classifierNames]["Model"])
        classifierList.append(classifierTuple)

    # Create the voting classifier
    votingClassifier = VotingClassifier(estimators=classifierList, voting=svmVoting)
    if svmVoting == "soft":
        votingClassifier.named_estimators["SVM"].probability = True

    # Into the model dict:
    tempDict = {"Model": votingClassifier, "Params": []}
    modelDictionary["VotingClassifier"] = tempDict

# Create the classifier model:
currentModel = modelDictionary[modelType]["Model"]

# Fit the model to the training data:
trainLabels = datasets["train"]["labels"].values.ravel()  # From column to row
trainFeatures = datasets["train"]["dataset"]
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
_, stratAccuracy, stratStdDev = stratifiedCrossValidation(currentModel, trainFeatures, trainLabels,
                                                          randomState=rng, splits=10, testSize=0.2,
                                                          verbose=False)

if testSplit != -1:

    # Test current model:
    print("[INFO] --- Testing Classifier...")
    testLabels = datasets["validation"]["labels"].values.ravel()  # From column to row
    testFeatures = datasets["validation"]["dataset"]

    if modelType == "VotingClassifier":
        print("Computing Voting Accuracies...")
        for classifierName, currentClassifier in modelDictionary[modelType]["Model"].named_estimators_.items():
            print(" ", classifierName, ":", currentClassifier.score(testFeatures, testLabels))

    modelScore = currentModel.score(testFeatures, testLabels)
    print(">> Accuracy:", modelScore)

    print("[INFO] --- Computing Per sample accuracy...")

    # Get per sample accuracy:
    realClasses = perSampleAccuracy(testFeatures, testLabels, currentModel)

    # Get and display results:
    displayResults(currentModel, testFeatures, testLabels, realClasses, cvMean, cvStdDev, 10)
    # Display Stratified CV results:
    print("Strat. Mean Accuracy:", "{:.4f}".format(stratAccuracy), "Strat. Std.Dev:", "{:.4f}".format(stratStdDev))

    # Plot confusion matrix:
    plotConfusionMatrix(testLabels, currentModel)

# Check parameters for grid search:
gridParameters = modelDictionary[modelType]["Params"]
# Grid search:
if runGridSearch and gridParameters:
    # Hyperparameter optimization using Grid Search:
    print("[INFO] --- Running Grid Search for: ", modelType)

    optimizedModels = {"SVM": svm.SVC(), "LogisticRegression": LogisticRegression(max_iter=400)}

    # currentModel = LogisticRegression(max_iter=400)
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

    # Get per sample accuracy:
    realClasses = perSampleAccuracy(testFeatures, testLabels, optimizedModel)

    # Get and display results:
    displayResults(optimizedModel, testFeatures, testLabels, realClasses, cvMean, cvStdDev, 3)

    # Plot confusion matrix:
    plotConfusionMatrix(testLabels, optimizedModel)

print(">> Computing final predictions...")

# Fit the model to the test data:
testFeatures = datasets["test"]["dataset"]
finalPredictions = currentModel.predict(testFeatures)

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
    print(">> Writing output file: ", outFilename)
    passengerList.to_csv(outFilename, index=False)

print(">> Done. Fuck you.")

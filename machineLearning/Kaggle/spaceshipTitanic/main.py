
# File        :   spaceshipTitanic.py
# Version     :   1.0.0
# Description :   Solution for Kaggle's Spaceship Titanic problem
#                 (https://www.kaggle.com/competitions/spaceship-titanic)

# Date:       :   Nov 6, 2023
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

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_score, \
    recall_score
from sklearn.metrics import precision_recall_curve

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


# Project Path:
projectPath = "D://dataSets//spaceTitanic//"
# File Names:
datasetNames = ["train", "validation", "test"]

# Dataset file extension:
fileExtension = ".csv"

# Prediction label:
predictionLabel = "Transported"

# Script options
randomSeed = 42
numericalBins = 5
runGridSearch = True

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

for d, datasetName in enumerate(datasetNames):

    print("Processing Dataset:", datasetName)

    # Select dataset type:
    if datasetName == "train":

        print("Splitting training + validation datasets...")

        # Read the cvs file:
        currentDataset = pd.read_csv(projectPath + datasetName + fileExtension)

        # Split the training dataset into train + validation:
        trainDataset, validationDataset = train_test_split(currentDataset, test_size=0.2, random_state=randomSeed)

        # Store the validation dataset:
        datasets["validation"]["dataset"] = validationDataset

        # Set the dataset to be processed:
        currentDataset = trainDataset

    elif datasetName == "validation":

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
        print("Dropping column: ", currentFeature)
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

        # Get class distribution:
        classCounter = Counter(classLabels)

        # Get total entries:
        totalEntries = len(predictionTarget)

        # Print the class distribution:
        for c in classCounter:
            # Get counter value
            value = classCounter[c]
            percent = format(100 * (value / totalEntries), ".2f")
            # Print distribution:
            print("Class:", c, "count:", value, "{ " + str(percent) + "%" + " }")

    # NaN Replacement of the following features:
    targetFeatures = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
    for currentFeature in targetFeatures:
        # Replace "NaN" with "NA-FEATURE":
        replacementString = "NA-" + currentFeature
        print("Replacing NaNs in: ", currentFeature, "with", replacementString)
        currentDataset[currentFeature] = replaceFeatureValue(currentDataset[currentFeature], np.NaN, replacementString)
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
        currentImputer = SimpleImputer(missing_values=np.NaN, strategy="mean")
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
            # Get Interquartile Range between Q1 and Q3:
            Q1 = numericalFeatures[currentFeature].quantile(0.25)
            Q3 = numericalFeatures[currentFeature].quantile(0.75)

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

    # Store in dataset dictionary:
    datasets[datasetName]["dataset"] = preprocessedDataset

    print("Finished preprocessing for dataset: ", datasetName)

# Fit the model:
print("Fitting model...")

# Create the logistic reggresor model:
logisticReggressor = LogisticRegression(solver="liblinear", C=1.0, random_state=42)

# Fit the model to the training data:
trainLabels = datasets["train"]["labels"].values.ravel()  # From column to row
trainFeatures = datasets["train"]["dataset"]
logisticReggressor.fit(trainFeatures, trainLabels)

# Cross-validation folds:
cvFolds = 10
# Cross-validation parallel jobs (1 per core):
parallelJobs = 5

# Check out the reggressor accuracy using cross-validation:
print("[INFO] --- Cross-Validating Classifier...")
reggressorAccuracy = cross_val_score(estimator=logisticReggressor, X=trainFeatures, y=trainLabels, cv=cvFolds,
                                     n_jobs=parallelJobs, verbose=3)

# Accuracy for each fold:
print("[INFO] --- Fold Accuracy:")
print(" ", reggressorAccuracy)
print("[INFO] --- Mean & Std Dev Fold Accuracy:")
print(">> Mu: ", np.mean(np.array(reggressorAccuracy)), "Sigma:", np.std(np.array(reggressorAccuracy)))

# Test the Regression Model:
print("[INFO] --- Testing Classifier...")
testLabels = datasets["validation"]["labels"].values.ravel()  # From column to row
testFeatures = datasets["validation"]["dataset"]

predictionProbabilities = logisticReggressor.predict_proba(testFeatures)
regressorPredictions = logisticReggressor.predict(testFeatures)

regressorScore = logisticReggressor.score(testFeatures, testLabels)
print(">> Accuracy:", regressorScore)

print("[INFO] --- Per sample accuracy...")

# Real classes counter:
realClasses = {"0": 0, "1": 0}

# Print the predicted class, real class and probabilities per sample:
for i in range(len(testFeatures)):
    # Get sample max probability:
    sampleProbability = np.max(predictionProbabilities[i])
    # Get predicted class:
    sampleClass = np.argmax(predictionProbabilities[i])
    # Get real class:
    realClass = testLabels[i]
    # Into class counter:
    realClasses[str(realClass)] += 1
    # Print missmatch:
    missmatch = ""
    if realClass != sampleClass:
        missmatch = " <-"
    # Print the info:
    print(" Sample:", i, "Truth:", realClass, "Predicted:", sampleClass,
          "(Proba: " + "{:.4f}".format(sampleProbability) + ")" + missmatch)

# Get confusion matrix and its plot:
print("[INFO] --- Plotting CM")

result = confusion_matrix(testLabels, regressorPredictions)  # normalize='pred'
print(result)
accuracy = (result[0][0] + result[1][1]) / len(testLabels)
print(realClasses)

# Compute precision & recall:
modelPrecision = precision_score(testLabels, regressorPredictions)
modelRecall = recall_score(testLabels, regressorPredictions)
# Print the results:
dateNow = time.strftime("%Y-%m-%d %H:%M")
print("---------------------------------------------------------- ")
print("Results Test time: " + dateNow)
print("Precision: ", modelPrecision)
print("Recall: ", modelRecall)
print("---------------------------------------------------------- ")
print("Test CF:", accuracy)
print("Validation Accuracy:", regressorScore)
print("Cross-validation Mean: ", np.mean(np.array(reggressorAccuracy)))

confusionMatrix = confusion_matrix(testLabels, regressorPredictions, labels=logisticReggressor.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=logisticReggressor.classes_)
disp.plot()
plt.show()

# Grid search:
if runGridSearch:
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
    print("[INFO] --- [Post-Grid Search] Cross-Validating Classifier...")
    reggressorAccuracy = cross_val_score(estimator=logisticReggressorOptimized, X=trainFeatures, y=trainLabels,
                                         cv=cvFolds,
                                         n_jobs=parallelJobs, verbose=3)

    # Accuracy for each fold:
    print("[INFO] --- [Post-Grid Search] Fold Accuracy:")
    print(" ", reggressorAccuracy)
    print("[INFO] --- [Post-Grid Search] Mean & Std Dev Fold Accuracy:")
    print(">> Mu: ", np.mean(np.array(reggressorAccuracy)), "Sigma:", np.std(np.array(reggressorAccuracy)))

    print("Fuck You")

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


# # One-hot encodes feature 0 and 2 (categorical -> Deck, Side) and
# # Adds Imputes and Scales feature 1 (numerical -> Cabin Number)
# def processCabinFeature(dataframe):
#     # Dataframe shallow copy:
#     outDataframe = dataframe
#     # Encoders list:
#     cabinEncoders = []
#     # For the three sub-features created before:
#     for i in range(3):
#         if i != 1:
#             # Change NaNs for "NA-Cabin" + str(i)
#             replacementString = "NA-Cabin-" + str(i)
#             outDataframe[[i]] = replaceFeatureValue(outDataframe[[i]], np.NaN, replacementString)
#
#             # Create encoder object and apply it to the dataframe:
#             currentEncoder = OneHotEncoder()
#             currentEncoded = currentEncoder.fit_transform(outDataframe[[i]])
#             currentEncoded = currentEncoded.toarray()
#
#             # Check out categories:
#             print(currentEncoder.categories_)
#
#             # Attach/append to outDataframe:
#             outDataframe[currentEncoder.categories_[0]] = currentEncoded
#             # Append to encoders list:
#             cabinEncoders.append((currentEncoder, "Encoder: " + str(i)))
#         else:
#             # Impute missing values with median:
#             tempFeature = outDataframe[[i]]
#             currentImputer = SimpleImputer(missing_values=np.NaN, strategy="median")
#
#             # Fit + transform transformer:
#             tempFeature = currentImputer.fit_transform(tempFeature)
#
#             # Scale feature:
#             currentScaler = StandardScaler()
#             tempFeature = currentScaler.fit_transform(tempFeature)
#
#             # Attach/append to outDataframe:
#             outDataframe["CabinNum"] = tempFeature
#
#             # Append to encoders list:
#             cabinEncoders.append((currentImputer, "Imputer: " + str(i)))
#             cabinEncoders.append((currentScaler, "Scaler: " + str(i)))
#
#     # Produce the final dataset slicing the temp dataset.
#     # Slice from the new columns to the ends, include all rows:
#     outDataframe = outDataframe.iloc[:, 3:]
#
#     # Return the list of encoders + outDataframe:
#     return cabinEncoders, outDataframe


# # Pre-processes numerical features:
# def processNumericalFeatures(inputDataFrame, bins=5):
#     outParameters = {}
#     # Slice numerical features:
#     numericalFeatures = inputDataFrame.loc[:, "RoomService":"VRDeck"]
#     featureNames = numericalFeatures.columns.values.tolist()
#
#     currentImputer = SimpleImputer(missing_values=np.NaN, strategy="mean")
#
#     # Fit + transform transformer:
#     numericalFeatures = currentImputer.fit_transform(numericalFeatures)
#
#     # isolation_forest = IsolationForest(random_state=42)
#     # outlier_pred = isolation_forest.fit_predict(numericalFeatures)
#
#     # Convert array to data frame:
#     numericalFeatures = pd.DataFrame(data=numericalFeatures, columns=featureNames)
#
#     # numericalFeatures2 = numericalFeatures.iloc[outlier_pred == 1]
#
#     # binLabels = ["No Billed", "Low Billed", "Mid Billed", "High Billed"]
#     binLabels = list(range(bins))
#
#     cleanedOutliers = pd.DataFrame()
#
#     for currentFeature in featureNames:
#         # Get Interquartile Range between Q1 and Q3:
#         Q1 = numericalFeatures[currentFeature].quantile(0.25)
#         Q3 = numericalFeatures[currentFeature].quantile(0.75)
#         # Get IQR:
#         IQR = Q3 - Q1
#         # Compute lower and higher thresholds:
#         whisker_width = 1.5
#         lowerWhisker = Q1 - (whisker_width * IQR)
#         upperWhisker = Q3 + (whisker_width * IQR)
#
#         # Replace all the values that are below the 25th percentile and above the 75th percentile of
#         # the current feature:
#         cleanedOutliers[currentFeature] = np.where(numericalFeatures[currentFeature] > upperWhisker, upperWhisker,
#                                                    np.where(numericalFeatures[currentFeature] < lowerWhisker,
#                                                             lowerWhisker, numericalFeatures[currentFeature]))
#
#         # Segment feature into len(binLabels) bins::
#         totalBins = len(binLabels)
#         encodedFeature = pd.cut(cleanedOutliers[currentFeature], bins=totalBins, labels=binLabels)
#
#         # Append new feature:
#         numericalFeatures[currentFeature + "-Bined"] = encodedFeature
#
#         # Add to out parameters dict:
#         if currentFeature not in outParameters:
#             outParameters[currentFeature] = {}
#         outParameters[currentFeature]["Thresholds"] = (lowerWhisker, upperWhisker)
#
#     # Get the binned features only:
#     cleanedOutliers = numericalFeatures.loc[:, "RoomService-Bined":"VRDeck-Bined"]
#
#     # Scale feature:
#     currentScaler = StandardScaler()
#     cleanedOutliers = currentScaler.fit_transform(cleanedOutliers)
#
#     outParameters["Scaler"] = currentScaler
#
#     cleanedOutliers = pd.DataFrame(data=cleanedOutliers, columns=featureNames)
#
#     # Done
#     return outParameters, cleanedOutliers


# Project Path:
projectPath = "D://dataSets//spaceTitanic//"
# File Names:
datasetNames = ["train.csv", "test.csv"]
# Prediction label:
predictionLabel = "Transported"

# Script options
randomSeed = 42
numericalBins = 5

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

for d, datasetName in enumerate(datasetNames):

    # Read the cvs file:
    currentDataset = pd.read_csv(projectPath + datasetName)

    # Remove extension from dataset name string:
    datasetName = datasetName[:-4]
    print("Processing Dataset:", datasetName)

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
    if datasetName == "train":
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

        # Store predicting column:
        predictionTargets[datasetName] = predictionTarget

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
            # Create the encoder object:
            currentEncoder = OneHotEncoder()
            # Fit + transform to feature:
            encodedFeature = currentEncoder.fit_transform(currentDataset[[currentFeature]])

            # Store encoder into dictionary:
            if currentFeature not in encodersDictionary:
                encodersDictionary[currentFeature] = currentEncoder

        else:
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
                # Create encoder object and apply it to the dataframe:
                currentEncoder = OneHotEncoder()
                currentEncoded = currentEncoder.fit_transform(tempDataframe[[i]])

                # Store encoder into dictionary:
                if featureString not in encodersDictionary:
                    encodersDictionary[featureString] = currentEncoder

            else:
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
        # Set imputer,
        # Maybe the strategy here could be median or most frequent, despite both values being 0:
        currentImputer = SimpleImputer(missing_values=np.NaN, strategy="mean")
        # Fit + transform transformer:
        numericalFeatures = currentImputer.fit_transform(numericalFeatures)

        # Store imputer into dictionary:
        if featureString not in encodersDictionary:
            encodersDictionary[featureString + "-Imputer"] = currentImputer

    else:

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
        # Set scaler:
        currentScaler = StandardScaler()
        # Fit + transform:
        tempDataframe = currentScaler.fit_transform(tempDataframe)

        # Store scaler into dictionary:
        if featureString not in encodersDictionary:
            encodersDictionary[featureString + "-Scaler"] = currentScaler

    else:
        # Set scaler:
        currentScaler = encodersDictionary[featureString + "-Scaler"]
        # Transform:
        tempDataframe = currentScaler.transform(tempDataframe)

    # Prepare the final dataframe (processed data + column names):
    tempDataframe = pd.DataFrame(data=tempDataframe, columns=featureNames)

    # Append/Concat to original dataframe based on left indices:
    preprocessedDataset = preprocessedDataset.join(tempDataframe)

    print("Finished preprocessing for dataset: ", datasetName)

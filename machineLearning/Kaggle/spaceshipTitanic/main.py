import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

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


# One-hot encodes feature 0 and 2 (categorical -> Deck, Side) and
# Adds Imputes and Scales feature 1 (numerical -> Cabin Number)
def processCabinFeature(dataframe):
    # Dataframe shallow copy:
    outDataframe = dataframe
    # Encoders list:
    cabinEncoders = []
    # For the three sub-features created before:
    for i in range(3):
        if i != 1:
            # Change NaNs for "NA-Cabin" + str(i)
            replacementString = "NA-Cabin-" + str(i)
            outDataframe[[i]] = replaceFeatureValue(outDataframe[[i]], np.NaN, replacementString)

            # Create encoder object and apply it to the dataframe:
            currentEncoder = OneHotEncoder()
            currentEncoded = currentEncoder.fit_transform(outDataframe[[i]])
            currentEncoded = currentEncoded.toarray()

            # Check out categories:
            print(currentEncoder.categories_)

            # Attach/append to outDataframe:
            outDataframe[currentEncoder.categories_[0]] = currentEncoded
            # Append to encoders list:
            cabinEncoders.append((currentEncoder, "Encoder: " + str(i)))
        else:
            # Impute missing values with median:
            tempFeature = outDataframe[[i]]
            currentImputer = SimpleImputer(missing_values=np.NaN, strategy="median")

            # Fit + transform transformer:
            tempFeature = currentImputer.fit_transform(tempFeature)

            # Scale feature:
            currentScaler = StandardScaler()
            tempFeature = currentScaler.fit_transform(tempFeature)

            # Attach/append to outDataframe:
            outDataframe["CabinNum"] = tempFeature

            # Append to encoders list:
            cabinEncoders.append((currentImputer, "Imputer: " + str(i)))
            cabinEncoders.append((currentScaler, "Scaler: " + str(i)))

    # Produce the final dataset slicing the temp dataset.
    # Slice from the new columns to the ends, include all rows:
    outDataframe = outDataframe.iloc[:, 3:]

    # Return the list of encoders + outDataframe:
    return cabinEncoders, outDataframe


# Pre-processes numerical features:
def processNumericalFeatures(inputDataFrame, bins=5):
    outParameters = {}
    # Slice numerical features:
    numericalFeatures = inputDataFrame.loc[:, "RoomService":"VRDeck"]
    featureNames = numericalFeatures.columns.values.tolist()

    currentImputer = SimpleImputer(missing_values=np.NaN, strategy="mean")

    # Fit + transform transformer:
    numericalFeatures = currentImputer.fit_transform(numericalFeatures)

    # isolation_forest = IsolationForest(random_state=42)
    # outlier_pred = isolation_forest.fit_predict(numericalFeatures)

    # Convert array to data frame:
    numericalFeatures = pd.DataFrame(data=numericalFeatures, columns=featureNames)

    # numericalFeatures2 = numericalFeatures.iloc[outlier_pred == 1]

    # binLabels = ["No Billed", "Low Billed", "Mid Billed", "High Billed"]
    binLabels = list(range(bins))

    cleanedOutliers = pd.DataFrame()

    for currentFeature in featureNames:
        # Get Interquartile Range between Q1 and Q3:
        Q1 = numericalFeatures[currentFeature].quantile(0.25)
        Q3 = numericalFeatures[currentFeature].quantile(0.75)
        # Get IQR:
        IQR = Q3 - Q1
        # Compute lower and higher thresholds:
        whisker_width = 1.5
        lower_whisker = Q1 - (whisker_width * IQR)
        upper_whisker = Q3 + (whisker_width * IQR)

        # Replace all the values that are below the 25th percentile and above the 75th percentile of
        # the current feature:
        cleanedOutliers[currentFeature] = np.where(numericalFeatures[currentFeature] > upper_whisker, upper_whisker,
                                                   np.where(numericalFeatures[currentFeature] < lower_whisker,
                                                            lower_whisker, numericalFeatures[currentFeature]))

        # Segment feature into len(binLabels) bins::
        totalBins = len(binLabels)
        encodedFeature = pd.cut(cleanedOutliers[currentFeature], bins=totalBins, labels=binLabels)

        # Append new feature:
        numericalFeatures[currentFeature + "-Bined"] = encodedFeature

        # Add to out parameters dict:
        if currentFeature not in outParameters:
            outParameters[currentFeature] = {}
        outParameters[currentFeature]["Thresholds"] = (lower_whisker, upper_whisker)

    # Get the binned features only:
    cleanedOutliers = numericalFeatures.loc[:, "RoomService-Bined":"VRDeck-Bined"]

    # Scale feature:
    currentScaler = StandardScaler()
    cleanedOutliers = currentScaler.fit_transform(cleanedOutliers)

    outParameters["Scaler"] = currentScaler

    cleanedOutliers = pd.DataFrame(data=cleanedOutliers, columns=featureNames)

    # Done
    return outParameters, cleanedOutliers


# Project Path:
projectPath = "D://dataSets//spaceTitanic//"
# File Names:
datasetName = "train.csv"
# Prediction label:
predictionLabel = "Transported"

randomSeed = 42

# Set console format:
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

# Set random seed:
random.seed(randomSeed)

# Read the dataset:
inputDataset = pd.read_csv(projectPath + datasetName)
# Print dataset shape:
print(inputDataset)

# Get a quick description of the data:
print(inputDataset.info())

# Check out numerical features only:
print(inputDataset.describe())

# Drop PassengerId and Name columns:
dropFeatures = ["PassengerId", "Name"]
for currentFeature in dropFeatures:
    print("Dropping colum: ", currentFeature)
    inputDataset = inputDataset.drop(currentFeature, axis=1)

# Replace "Transported" Feature with 0 (False) or 1 (True):
print("[INFO] --- Setting Predictive Feature...")
inputDataset[predictionLabel] = replaceFeatureValue(inputDataset[predictionLabel], True, 1)
inputDataset[predictionLabel] = replaceFeatureValue(inputDataset[predictionLabel], False, 0)

# Get feature names:
featureNames = inputDataset.columns.values.tolist()

# Split train & validation datasets:
print("[INFO] --- Splitting Complete Dataset...")
trainDataset, validationDataset = train_test_split(inputDataset, test_size=0.2, random_state=randomSeed)
trainDatasetCopy = trainDataset.copy()

# Get the training features and target feature/predicting label:
trainTarget = pd.DataFrame(data=trainDataset[predictionLabel], columns=[predictionLabel])
trainTarget = trainTarget.reset_index(drop=True)
trainDataset = trainDataset.drop(predictionLabel, axis=1)

# NaN Replacement of the following features:
targetFeatures = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
for currentFeature in targetFeatures:
    # Replace "NaN" with "NA-FEATURE":
    replacementString = "NA-" + currentFeature
    print("Replacing NaNs in: ", currentFeature, "with", replacementString)
    trainDataset[currentFeature] = replaceFeatureValue(trainDataset[currentFeature], np.NaN, replacementString)
    # Additionally, replace "True" and "False" in "CryoSleep/VIP" Column:
    if currentFeature == "CryoSleep" or currentFeature == "VIP":
        print("Replacing True/False in: ", currentFeature)
        trainDataset[currentFeature] = replaceFeatureValue(trainDataset[currentFeature], True, currentFeature + "-TRUE")
        trainDataset[currentFeature] = replaceFeatureValue(trainDataset[currentFeature], False,
                                                           currentFeature + "-FALSE")

# Process "Age" Feature:
# Replace "NaN" with "-0.5":
trainDataset["Age"] = replaceFeatureValue(trainDataset[["Age"]], np.NaN, -0.5)

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
labeledAge = pd.cut(trainDataset["Age"], bins=ageBins, labels=binLabels)

# Convert series to data frame:
labeledAge = pd.DataFrame(labeledAge)

# Store in original dataset:
trainDataset["Age-Labeled"] = labeledAge

# Directly One-hot encode categorical features:
categoricalFeatures = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Age-Labeled"]

# Prepare the final preprocessed dataset:
preprocessedDataset = pd.DataFrame()

# Prepare the dictionary of encoders:
encodersDictionary = {}

for currentFeature in categoricalFeatures:
    print("One-hot encoding feature:", currentFeature)
    # Create the encoder object:
    currentEncoder = OneHotEncoder()
    # Fit + transform to feature:
    encodedFeature = currentEncoder.fit_transform(trainDataset[[currentFeature]])
    # Convert to array:
    encodedFeature = encodedFeature.toarray()

    # Print categories:
    print("Categories: ", currentEncoder.categories_)

    # Append to categorical dataset:
    preprocessedDataset[currentEncoder.categories_[0]] = encodedFeature

    # Store encoder into dictionary:
    if currentFeature not in encodersDictionary:
        encodersDictionary[currentFeature] = currentEncoder

# Process "Cabin" Feature:
# Split cabin feature into 3 sub-features:
cabinSplit = trainDataset["Cabin"].str.split("/", expand=True)

# For every new cabin sub-feature, one hot encode it except the
# Cabin Number feature (column 1). Store all the encoders used
cabinEncoders, cabinSplit = processCabinFeature(cabinSplit)

# Reset indices:
cabinSplit = cabinSplit.reset_index(drop=True)

# Append/Concat to original dataframe based on left indices:
# trainDatasetCopy = pd.merge(trainDatasetCopy, cabinSplit, left_index=True, right_index=True)
preprocessedDataset = preprocessedDataset.join(cabinSplit)

# Numerical features:
transformersUsed, numericalFeatures = processNumericalFeatures(trainDataset)

# Append/Concat to original dataframe based on left indices:
preprocessedDataset = preprocessedDataset.join(numericalFeatures)

print("Fuck")

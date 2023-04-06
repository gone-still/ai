# File        :   customerClassification.py
# Version     :   0.9.1
# Description :   A "challenge" to apply NLP techniques to classify customer requests
#                 into products...

# Date:       :   Apr 05, 2023
# Author      :   Ricardo Acevedo-Avila
# License     :   Creative Commons CC0

import nltk, random
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV


# Applies basic word and character filtering to
# a document:
def wordFilter(inputDocuments, filteredWords):
    # Word filter:
    # All strings to lowercase, removes punctuation, removes "xxxxx" and
    # removes more than one whitespaces in messages:
    for m in range(len(inputDocuments)):
        currentMessage = inputDocuments[m]
        currentMessage = currentMessage.lower()
        # Regex to remove punctuation:
        currentMessage = re.sub(r"[^\w\s]", '', currentMessage)

        # Apply filter word. Replaces targets with empty string:
        for f in range(len(filteredWords)):
            targetWord = filteredWords[f]
            currentMessage = currentMessage.replace(targetWord, " ")

        # Removes more than one whitespace character:
        currentMessage = re.sub(r"\s\s+", " ", currentMessage)

        inputDocuments[m] = currentMessage

    # Done, return the filtered list of documents:
    return inputDocuments


# Load the dataset:
filePath = "D://dataSets//customer-issues-train-03.csv"
nlpDataset = pd.read_csv(filePath)

# Check out the columns:
print(nlpDataset.columns)

# Check out the first 5 samples:
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)
print(nlpDataset.head())

# Drop missing values:
nlpDataset = nlpDataset.dropna(axis=0)
print("-----------------------------")
print(nlpDataset.head())

# Extract categorical columns:
nlpDataset_columns = nlpDataset.copy()
nlpDataset_columns = nlpDataset_columns[["product", "sub-product", "issue", "resolution"]]
print("-----------------------------")
print(nlpDataset_columns.head())

# Process predicting class/feature:
classLabels = nlpDataset_columns["product"]
classCounter = Counter(classLabels)

print(" ------ Class distribution ------ ")
for c in classCounter:
    value = classCounter[c]
    print("Class: ", c, " Count: ", value)

# Remove duplicated class labels:
classesDict = classLabels.drop_duplicates()

# Reset index, store as a dictionary:
classesDict = classesDict.reset_index(drop=True)
classesDict = classesDict.to_dict()

# Loop through the classes dictionary and
# encode each class label as a number:
for i, key in enumerate(classesDict):
    currentClass = classesDict[key]
    classLabels = np.where(classLabels == currentClass, key, classLabels)

# Array to data frame:
classLabels = pd.DataFrame(classLabels, columns=["class"])
# Check out the class encoding:
f = nlpDataset_columns["product"].reset_index(drop=True)
encodedTable = pd.concat([f, classLabels], axis=1)

print("---- Class encoding -----------------------------")
print(encodedTable)

# Process categorical features:
features = ["sub-product", "issue", "resolution"]
totalFeatures = len(features)
# Percent of the label values to preserve:
labelsPercent = 0.8

# Encoded binary dataset:
encodedDataset = pd.DataFrame()

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 100)

# One-hot encode top 80% of the categorical features:
for i, f in enumerate(features):
    # Get current feature:
    currentFeature = features[i]
    # Get count of unique values for this feature:
    labelCount = len(nlpDataset_columns[f].unique())

    # Get labels percent:
    filteredCount = int(labelsPercent * labelCount)
    print("---- Feature Stats -----------------------------")
    print("Feature:", f, ":", "Values:", labelCount, "Filtered:", filteredCount)

    # Print filtered feature values:
    print(nlpDataset_columns[f].value_counts().sort_values(ascending=False))

    # Get filtered labels:
    filteredLabels = [x for x in nlpDataset_columns[f].value_counts().sort_values(ascending=False).head(
        filteredCount).index]

    # The top labels with 80% of the count:
    print("---- Filtered labels -----------------------------")
    print("Filtered labels:", filteredLabels)

    # For each filtered label, encode it via one-hot encoding:
    for label in filteredLabels:
        # Cell is set to 1 if the current filtered label is found on the current
        # row/sample. Cell is set to 0 otherwise:
        nlpDataset_columns[label] = np.where(nlpDataset_columns[f] == label, 1, 0)

    # Concatenate encoded feature to new data frame, along the horizontal axis (add columns):
    encodedDataset = pd.concat([encodedDataset, nlpDataset_columns[filteredLabels]], axis=1)
    # Produce a local data frame to check out the encodings:
    encodedTable = nlpDataset_columns[[f] + filteredLabels]
    print(encodedTable)

# Reset the data frame indices:
encodedDataset = encodedDataset.reset_index(drop=True)
# Show the final binary dataset:
print("-----------------------------")
print(encodedDataset.head(5))

# Use Bag of Words to vectorize the "consumer-message" feature:
wobFeature = nlpDataset.copy()
datasetDocuments = wobFeature["consumer-message"].values.tolist()

# Print the dataset documents:
print("-------------- Dataset Documents ---------------")
print(datasetDocuments)

# Apply the word filter before message vectorization:
filteredWords = ["xxxx", " i ", "i ", " us ", " we ", " am "]
datasetDocuments = wordFilter(datasetDocuments, filteredWords)

# List to dataframe:
datasetDocuments = pd.DataFrame({"consumer-message": datasetDocuments})
# Create complete dataset (Encoded features + documents + classes):
completeDataset = pd.concat([classLabels, encodedDataset, datasetDocuments], axis=1)
# completeDataset = pd.concat([classLabels, datasetDocuments], axis=1)

# Dataset division, training: 80% testing: 20%
trainDataset, testDataset = train_test_split(completeDataset, test_size=0.20, random_state=42069)

# Reset dataframe indices:
trainDataset = trainDataset.reset_index(drop=True)
testDataset = testDataset.reset_index(drop=True)

# Sample/Text vectorization:
# Extract only the test and train documents:
trainDocuments = trainDataset["consumer-message"]
testDocuments = testDataset["consumer-message"]

# Use bag of words to vectorize each sample:
# Bag of words is applied via the TfidfVectorizer object,
# Each token is a word with the characters a-z or A-Z,
# min_df is the minimal threshold of word frequency needed to
# not filter out the word:
wordVectorizer = TfidfVectorizer(min_df=0.1, max_df=0.8, token_pattern=r'[a-zA-Z]+')

# Fit Train:
trainSamplesVectorized = wordVectorizer.fit_transform(trainDocuments)
# Fit Test:
testSamplesVectorized = wordVectorizer.transform(testDocuments)

# Check out the vectorized dataset. Vector size (second dimension)
print("Train Samples shape: ", trainSamplesVectorized.shape)
print("Test Samples shape: ", testSamplesVectorized.shape)

# Sparse matrices into pandas dataframe:
trainDocumentEncoding = pd.DataFrame(trainSamplesVectorized.toarray(), columns=wordVectorizer.get_feature_names_out())
testDocumentEncoding = pd.DataFrame(testSamplesVectorized.toarray(), columns=wordVectorizer.get_feature_names_out())

# Check out the document encoding (per sample) on the training dataset:
print(trainDocumentEncoding)

# Create Complete Datasets, replace "consumer-message" feature with vectorized encoding:
trainDataset = trainDataset.drop("consumer-message", axis=1)
trainDataset = pd.concat([trainDataset, trainDocumentEncoding], axis=1)

testDataset = testDataset.drop("consumer-message", axis=1)
testDataset = pd.concat([testDataset, testDocumentEncoding], axis=1)

# Conversion to lists:
trainingFeatures = np.asarray(trainDataset.iloc[:, 1:].values.tolist())
trainingLabels = np.asarray(trainDataset.iloc[:, 0:1].values.tolist())

testFeatures = np.asarray(testDataset.iloc[:, 1:].values.tolist())
testLabels = np.asarray(testDataset.iloc[:, 0:1].values.tolist())

# Flatten the labels, because sklearn's SVM receives a row instead
# of a column for its class labels:
trainingLabels = np.ravel(trainingLabels)
testLabels = np.ravel(testLabels)

print("---- Creating and Fitting SVM  -----------------------------")
# Classification via SVM:
# Train the SVM
# SVM hyperparameters: C = 0.8 with linear kernel:
svmModel = svm.SVC(C=4.0, kernel="linear")
svmModel.fit(trainingFeatures, trainingLabels)

print("---- Cross Validating SVM  ---------------------------------")
# Evaluate SVM using cross validation, use 5 folds:
cvFolds = 5
svmAccuracy = cross_val_score(estimator=svmModel, X=trainingFeatures, y=trainingLabels, cv=cvFolds, n_jobs=-1)
# Accuracy for each fold:
print(svmAccuracy)

# Test the SVM:
svmPredictions = svmModel.predict(testFeatures)
print(svmModel.score(testFeatures, testLabels))

# Get confusion matrix and its plot:
confusionMatrix = confusion_matrix(testLabels, svmPredictions, labels=svmModel.classes_, normalize="all")
disp = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=svmModel.classes_)
disp.plot()
plt.show()

# print("---- Running Search Grid  ---------------------------------")
# # Hyperparameter optimization using Grid Search:
# parameters = {"kernel": ("linear", "rbf"), "C": (1, 4, 8, 16, 32)}
# svm = svm.SVC()
# optimizedSVM = GridSearchCV(svm, parameters, cv=5, n_jobs=-1)
# optimizedSVM.fit(trainingFeatures, trainingLabels)
#
# # Print hyperparameters & accuracy:
# print(optimizedSVM.best_params_)
# svmAccuracy = cross_val_score(estimator=optimizedSVM, X=trainingFeatures, y=trainingLabels, cv=cvFolds, n_jobs=-1)
# # Accuracy for each fold:
# print(svmAccuracy)

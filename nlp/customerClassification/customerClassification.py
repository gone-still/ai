# File        :   customerClassification.py
# Version     :   0.0.8
# Description :   An unpaid "challenge" from company X to apply NLP techniques
#                 to classify customer requests into products. The shitty dataset has
#                 already been processed through Weka's CfsSubsetEval to drop the 
#                 meaningless features and reduce dimentionality.

# Date:       :   Apr 04, 2023
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
trainDocuments = wobFeature["consumer-message"].values.tolist()

print(wobFeature)
print(trainDocuments)

# Word filter:
# All strings to lowercase, removes punctuation, removes "xxxxx" and
# removes more than one whitespaces in messages:
filteredWords = ["xxxx", " i ", "i ", " us ", " we ", " am "]
for m in range(len(trainDocuments)):
    currentMessage = trainDocuments[m]
    currentMessage = currentMessage.lower()
    # Regex to remove punctuation:
    currentMessage = re.sub(r"[^\w\s]", '', currentMessage)

    # Apply filter word. Replaces targets with empty string:
    for f in range(len(filteredWords)):
        targetWord = filteredWords[f]
        currentMessage = currentMessage.replace(targetWord, " ")

    # Removes more than one whitespace character:
    currentMessage = re.sub(r"\s\s+", " ", currentMessage)

    trainDocuments[m] = currentMessage

# Sample/Text vectorization:
# Use bag of words to vectorize each sample:
# Bag of words is applied via the TfidfVectorizer object,
# Each token is a word with the characters a-z or A-Z,
# min_df is the minimal threshold of word frequency needed to
# not filter out the word:
wordVectorizer = TfidfVectorizer(min_df=0.1, max_df=0.8, token_pattern=r'[a-zA-Z]+')

# Fit messages:
messagesVectorized = wordVectorizer.fit_transform(trainDocuments)

# Check out the vectorized dataset. Vector size (second dimension)
print("Vectorized messages shape: ", messagesVectorized.shape)

# Check out the word encoding per document/sample:
wordEncoding = pd.DataFrame(messagesVectorized.toarray(), columns=wordVectorizer.get_feature_names())

print(wordEncoding)

# Create Training Numerical Dataset:
trainingDataset = pd.concat([encodedDataset, wordEncoding], axis=1)

# Conversion to lists:
trainingDataset = np.asarray(trainingDataset.values.tolist())
trainingLabels = np.asarray(classLabels.values.tolist())
# Flatten the labels, because sklearn's SVM receives a row instead
# of a column for its class labels:
trainingLabels = np.ravel(trainingLabels)

print("---- Creating and Fitting SVM  -----------------------------")

# Classification via SVM:
# Train the SVM
# SVM hyperparameters: C = 0.8 with linear kernel:
svmModel = svm.SVC(C=1.0, kernel="linear")
svmModel.fit(trainingDataset, trainingLabels)

print("---- Cross Validating SVM  ---------------------------------")
# Evaluate SVM using cross validation, use 5 folds:
cvFolds = 5
svmAccuracy = cross_val_score(estimator=svmModel, X=trainingDataset, y=trainingLabels, cv=cvFolds, n_jobs=-1)
# Accuracy for each fold:
print(svmAccuracy)

# File        :   customerClassification-Train.py
# Version     :   1.1.1
# Description :   [Train + Test]
#                 An unpaid "challenge" from company X to apply NLP techniques
#                 to classify customer requests into products. The shitty dataset has
#                 already been pre-processed with Weka's CfsSubsetEval to drop the
#                 meaningless features and reduce dimensionality.

# Date:       :   Apr 13, 2023
# Author      :   Ricardo Acevedo-Avila
# License     :   Creative Commons CC0


import pandas as pd
import re

import matplotlib.pyplot as plt
import numpy as np

from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer

from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample

import seaborn as sns

import pickle


# Applies basic word and character filtering to
# a document:
def wordFilter(inputDocuments, tokenizer, stopwordList):
    # Word filter:
    # All strings to lowercase, removes punctuation, removes "xxxxx" and
    # removes more than one whitespaces in messages:
    for m in range(len(inputDocuments)):
        currentMessage = inputDocuments[m]
        currentMessage = currentMessage.lower()
        # Regex to remove punctuation:
        currentMessage = re.sub(r"[^\w\s]", '', currentMessage)

        # Removes XXX prefix:
        currentMessage = re.sub(r'^{0}'.format(re.escape("xxxx")), '', currentMessage)

        # Removes numbers:
        currentMessage = re.sub(" \d+", " ", currentMessage)

        # Apply filter word. Replaces targets with empty string:
        # for word in stopwordList:
        #     currentMessage = re.sub(r'\b%s\b' % word, '', currentMessage)

        tokens = tokenizer.tokenize(currentMessage)
        tokens = [token.strip() for token in tokens]
        filteredTokens = [token for token in tokens if token not in stopwordList]
        currentMessage = ' '.join(filteredTokens)

        # Removes more than one whitespace character:
        currentMessage = re.sub(r"\s\s+", " ", currentMessage)

        inputDocuments[m] = currentMessage

    # Done, return the filtered list of documents:
    return inputDocuments


# Performs dataset resampling:
def datasetResampling(inputDataset, downsampleSize):
    print(" datasetResampling>> Original Dataset shape: ", inputDataset.shape)
    print(" datasetResampling>> Resampling to: " + str(downsampleSize) + " samples per class.")

    tempDataset = pd.DataFrame()

    for i, key in enumerate(classesDict):
        # Get current class:
        currentClass = classesDict[key]

        # Get all the samples from current class:
        classFeatures = inputDataset[inputDataset["product"] == currentClass]

        currentLength = classFeatures.shape[0]
        print(" datasetResampling>> (Pre-resampling) Class: " + currentClass + " Samples:", currentLength)

        # Get the new class samples:
        classResampled = resample(classFeatures, replace=True, n_samples=downsampleSize, random_state=42069)

        currentLength = classResampled.shape[0]
        print(" datasetResampling>> (Post-resampling) Class: " + currentClass + " Samples:", currentLength)

        # Into the temp dataset:
        tempDataset = pd.concat([tempDataset, classResampled], axis=0)
        print(" datasetResampling>> Resample dataset samples:", tempDataset.shape[0])

    # Shuffle samples and reset indices:
    tempDataset = tempDataset.sample(frac=1).reset_index(drop=True)
    print(" datasetResampling>> Resampled Dataset shape (reshuffled): ", tempDataset.shape)

    # Done:
    return tempDataset


# One-hot encodes a set of features:
def encodeFeatures(inputDataset, features, labelsPercent):
    # Encoded binary dataset:
    encodedDataset = pd.DataFrame()

    # pd.set_option("display.max_rows", 500)
    # pd.set_option("display.max_columns", 100)
    # pd.set_option("display.width", 100)

    # One-hot encode top 80% of the categorical features:
    for i, f in enumerate(features):

        # Get current feature:
        currentFeature = features[i]
        # Get count of unique values for this feature:
        labelCount = len(inputDataset[f].unique())

        # Get labels percent:
        filteredCount = int(labelsPercent * labelCount)
        print(" encodeFeatures>> Feature Stats:")
        print("  Feature:", f, ":", "Values:", labelCount, "Filtered:", filteredCount)

        # Print filtered feature values:
        print(inputDataset[f].value_counts().sort_values(ascending=False))

        # Get filtered labels:
        filteredLabels = [x for x in inputDataset[f].value_counts().sort_values(ascending=False).head(
            filteredCount).index]

        # The top labels with 80% of the count:
        print(" encodeFeatures>> Filtered Labels:")
        print("  Filtered labels:", filteredLabels)

        # For each filtered label, encode it via one-hot encoding:
        for label in filteredLabels:
            # Cell is set to 1 if the current filtered label is found on the current
            # row/sample. Cell is set to 0 otherwise:
            inputDataset[label] = np.where(inputDataset[f] == label, 1, 0)

        # Concatenate encoded feature to new data frame, along the horizontal axis (add columns):
        encodedDataset = pd.concat([encodedDataset, inputDataset[filteredLabels]], axis=1)
        # Produce a local data frame to check out the encodings:
        encodedTable = inputDataset[[f] + filteredLabels]
        print(encodedTable)

    # Reset the data frame indices:
    encodedDataset = encodedDataset.reset_index(drop=True)
    # Show the final binary dataset:
    print("[INFO] --- Dataset 5 First Samples (One-hot Encoded):")
    print(encodedDataset.head(5))

    return encodedDataset


# Encodes documents into numerical vectors using BoW and
# prepares the final datasets as numpy arrays:
def encodeDocumentsPrepareDataset(datasetList, wordVectorizer):
    # Return list:
    outDatasets = []

    print("[INFO] --- Performing Document Encoding...")

    for d in range(len(datasetList)):
        # Get current Dataset:
        currentDataset = datasetList[d]

        # Reset dataframe indices:
        currentDataset = currentDataset.reset_index(drop=True)

        # Sample/Text vectorization:
        # Extract only the documents:
        currentDocuments = currentDataset["consumer-message"]

        if d == 0:
            print(" encodeDocuments>> Fit + Transforming training documents...")
            # Fit Train:
            samplesVectorized = wordVectorizer.fit_transform(currentDocuments)
        else:
            print(" encodeDocuments>> Transforming test documents...")
            # Fit Test:
            samplesVectorized = wordVectorizer.transform(currentDocuments)

        # Check out the vectorized dataset. Vector size (second dimension)
        print(" encodeDocuments>> Samples shape: ", samplesVectorized.shape)

        # Sparse matrices into pandas dataframe:
        documentEncoding = pd.DataFrame(samplesVectorized.toarray(), columns=wordVectorizer.get_feature_names_out())

        # Create Complete Datasets, replace "consumer-message" feature with vectorized encoding:
        currentDataset = currentDataset.drop("consumer-message", axis=1)
        currentDataset = pd.concat([currentDataset, documentEncoding], axis=1)

        # Conversion to numpy arrays for SVM consumption:
        currentFeatures = np.asarray(currentDataset.iloc[:, 1:].values.tolist())
        currentLabels = np.asarray(currentDataset.iloc[:, 0:1].values.tolist())

        # Flatten the labels, because sklearn's SVM receives a row instead
        # of a column for its class labels:
        currentLabels = np.ravel(currentLabels)

        # Into the list:
        outDatasets.append((currentFeatures, currentLabels))

    return outDatasets


# Script flags:
samplingFactor = 0.9
downloadStopword = False

fullFeatures = False
saveModel = True

crossValidateModel = True
parallelJobs = 5
runGridSearch = False

saveVocabulary = True
saveClassDictionary = True
saveVectorizer = True

cvFolds = 5

# Project Path:
projectPath = "D://dataSets//nlp-case//"

# File Names:
datasetName = "customer-issues-train-03.csv"
modelName = "customerClassifier-SVM.sav"
vocabularyName = "savedVocabulary.pkl"
dictionaryName = "classDict.pkl"
vectorizerName = "wordVectorizer.pkl"

# Read the CSV Dataset:
inputDataset = pd.read_csv(projectPath + datasetName)

# Check out the first 5 samples:
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

print("[INFO] --- Dataset 5 First Samples:")
print(inputDataset.head())

# Drop missing values:
print("[INFO] --- Columns with missing data:")
print(inputDataset.isnull().sum())
# inputDataset = inputDataset.dropna(axis=0)
print(inputDataset.isnull().sum())

print("[INFO] --- Dataset 5 First Samples (Dropped Missing Values):")
print(inputDataset.head())

# Extract categorical columns:
# Process predicting class/feature.
classLabels = inputDataset["product"]
classCounter = Counter(classLabels)

# List of class counts, stores the count of
# each class for median calculation:
classCounts = []

print("[INFO] --- Class Distribution:")
for c in classCounter:
    value = classCounter[c]
    print(" Class: ", c, " Count: ", value)
    # Into the class count list:
    classCounts.append(value)

# Plot class distribution:
inputDataset.groupby("product").size().plot(kind="pie", y="product", label="Type", autopct="%1.1f%%")
plt.show()

# Remove duplicated class labels:
classesDict = classLabels.drop_duplicates()

# Reset index, store as a dictionary:
classesDict = classesDict.reset_index(drop=True)
classesDict = classesDict.to_dict()
print("[INFO] --- The Classes Dictionary:")
for i, key in enumerate(classesDict):
    # Get current class:
    currentClass = classesDict[key]
    print(" " + str(key) + " --> " + str(currentClass))

# Get median of the count per class. This should be
# the class count to which all classes are resampled to...
classCounts = np.array(classCounts)
datasetMedian = np.median(classCounts)

# Loop through the classes dictionary and
# encode each class label as a number:
downsampleSize = samplingFactor * datasetMedian
downsampleSize = int(downsampleSize)

# Perform the dataset resampling:
print("[INFO] --- Resampling Dataset...")

# Reshuffle:
print(" Shuffling Dataset...")
inputDataset = inputDataset.sample(frac=1).reset_index(drop=True)
# Get resampled Dataset:
inputDataset = datasetResampling(inputDataset, downsampleSize)

# Plot the new class distribution:
inputDataset.groupby("product").size().plot(kind="pie", y="product", label="Type", autopct="%1.1f%%")
plt.show()

# Store "consumer-message" feature for bag of words encoding::
datasetDocuments = inputDataset["consumer-message"].values.tolist()
# Get class labels:
classLabels = inputDataset["product"]
classCounter = Counter(classLabels)

# Check out the new class distribution:
print("[INFO] --- Class Distribution (Resampled):")
for c in classCounter:
    value = classCounter[c]
    print(" Class: ", c, " Count: ", value)

# Encode classes:
print("[INFO] --- Encoding Classes...")
for i, key in enumerate(classesDict):
    # Get current class:
    currentClass = classesDict[key]
    classLabels = np.where(classLabels == currentClass, key, classLabels)

# Array to data frame:
classLabels = pd.DataFrame(classLabels, columns=["class"])

# Check out the class encoding:
encodedTable = pd.concat([inputDataset["product"].reset_index(drop=True), classLabels], axis=1)

print("[INFO] --- Encoded Classes Table:")
print(encodedTable)

# Process categorical features:
if fullFeatures:
    print("[INFO] --- Encoding Categorical Features...")
    inputDataset = inputDataset[["product", "sub-product", "issue", "resolution"]]

    print("[INFO] --- Dataset 5 First Samples (Categorical):")
    print(inputDataset.head())

    features = ["sub-product", "issue", "resolution"]
    totalFeatures = len(features)

    # Percent of the label values to preserve:
    labelsPercent = 0.8

    # Get the one-hot encoded dataset:
    inputDataset = encodeFeatures(inputDataset, features, labelsPercent)

# Print the dataset documents:
print("[INFO] --- Processing Dataset Documents...")
# print(datasetDocuments)

print("[INFO] --- Filtering Dataset Documents...")

# Download the stop words list:
if downloadStopword:
    nltk.download("stopwords")
# Tokenization of text
tokenizer = ToktokTokenizer()
# Setting English stopwords
stopwordList = nltk.corpus.stopwords.words("english")
# Add some custom words:
targetWords = ["xxxx", "xxxxxxxx", "xxxxxxxxxxxx", "company", "would", "told", "nt", "said", "could", "made", "still",
               "get", "since", "make", "help", "us", "please", "name", "going", "know", "name", "like", "customer",
               "one", "never", "take", "able", "also", "address", "stated", "new", "went", "need", "send", "time",
               "want", "got", "first", "husband", "wife", "see", "without", "last", "go", "however", "tried", "car",
               "took", "signed", "well", "way", "account", "accounts", "spoke", "put", "use", "give", "someone",
               "trying", "via", "thank", "though", "mine", "ask", "feel", "say", "per", "keep", "yet", "saying", "ca"]

# Add custom words to stopword list:
stopwordList.extend(targetWords)

print(" Stop Word List Length: " + str(len(stopwordList)))

# Apply the word filter before message vectorization:
datasetDocuments = wordFilter(datasetDocuments, tokenizer, stopwordList)

# Get word frequency:
wordsList = []
for line in datasetDocuments:
    words = line.split()
    for word in words:
        wordsList.append(word)

# Show word frequency:
sns.set_style('darkgrid')
nlp_words = nltk.FreqDist(wordsList)
nlp_words.plot(50)

# List to dataframe:
datasetDocuments = pd.DataFrame({"consumer-message": datasetDocuments})

# Create complete dataset (Encoded features + documents + classes):
if fullFeatures:
    print("[INFO] --- Created full Dataset (Labels + Encoded Features + Encoded Words)")
    completeDataset = pd.concat([classLabels, inputDataset, datasetDocuments], axis=1)

    # Free resources:
    del classLabels
    del inputDataset
    del datasetDocuments

else:
    print("[INFO] --- Created partial Dataset (Labels + Encoded Words)")
    completeDataset = pd.concat([classLabels, datasetDocuments], axis=1)

    # Free resources:
    del classLabels
    del datasetDocuments

# Dataset division, training: 80% testing: 20%
print("[INFO] --- Splitting Complete Dataset...")
trainDataset, testDataset = train_test_split(completeDataset, test_size=0.20, random_state=42069)

# Document encoding:
print("[INFO] --- Performing Document Encoding...")

# Use bag of words to vectorize each sample.
# Create and set the word vectorizer:
wordVectorizer = TfidfVectorizer(min_df=0.01, max_df=1.0, token_pattern=r'[a-zA-Z]+', sublinear_tf=True)

# Prepare the data structures:
datasetList = [trainDataset, testDataset]

# Get the document encodings:
outDatasets = encodeDocumentsPrepareDataset(datasetList, wordVectorizer)

# Classification via SVM:
# Train the SVM
print("[INFO] --- Creating and Fitting SVM...")
svmModel = svm.SVC(C=4.0, kernel="rbf")

# Get the training Dataset:
trainingFeatures = outDatasets[0][0]
trainingLabels = outDatasets[0][1]

svmModel.fit(trainingFeatures, trainingLabels)

# Evaluate SVM using cross validation, use 5 folds:
if crossValidateModel:
    print("[INFO] --- Cross-Validating SVM...")
    svmAccuracy = cross_val_score(estimator=svmModel, X=trainingFeatures, y=trainingLabels, cv=cvFolds,
                                  n_jobs=parallelJobs, verbose=3)
    # Accuracy for each fold:
    print(" Fold Accuracy:")
    print(" ", svmAccuracy)

# Test the SVM:
print("[INFO] --- Testing SVM...")

# Get the test Dataset:
testFeatures = outDatasets[1][0]
testLabels = outDatasets[1][1]

svmPredictions = svmModel.predict(testFeatures)
print(" SVM Accuracy:")
print(" ", svmModel.score(testFeatures, testLabels))

# Get confusion matrix and its plot:
print("[INFO] --- Plotting CM")
confusionMatrix = confusion_matrix(testLabels, svmPredictions, labels=svmModel.classes_, normalize="all")
disp = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=svmModel.classes_)
disp.plot()
plt.show()

# Save class dictionary:
if saveClassDictionary:
    dictionaryPath = projectPath + dictionaryName
    print("[INFO] --- Saving Class Dictionary to: " + dictionaryPath)
    pickle.dump(classesDict, open(dictionaryPath, "wb"))

# Save SVM model to disk
if saveModel:
    modelPath = projectPath + modelName
    print("[INFO] --- Saving SVM model to: ", projectPath + modelName)
    pickle.dump(svmModel, open(modelPath, "wb"))

# Save vocabulary:
if saveVocabulary:
    vocPath = projectPath + vocabularyName
    print("[INFO] --- Saving Vocabulary file to: " + vocPath)
    pickle.dump(wordVectorizer.vocabulary_, open(vocPath, "wb"))

# Save word vectorizer:
if saveVectorizer:
    vecPath = projectPath + "wordVectorizer.pkl"
    print("[INFO] --- Saving Word Vectorizer object to: " + vecPath)
    pickle.dump(wordVectorizer, open(vecPath, "wb"))

# Classify a new document:
print("[INFO] --- Classifying New Sample")

newSample = ["I obtained my free annual credit report online on XX/XX/2017. As of XX/XX/2017, a negative account is "
             "scheduled for deletion, I attempted to check and dispute online as of XXXX on XX/XX/2017 but was unable "
             "to do so, no matter what computer I used."]

# Vectorize new sample:
newSample = wordFilter(newSample, tokenizer, stopwordList)
newSampleVectorized = wordVectorizer.transform(newSample)
newSampleVectorized = pd.DataFrame(newSampleVectorized.toarray(), columns=wordVectorizer.get_feature_names_out())

# Dataframe to numpy array:
newSampleVectorized = newSampleVectorized.to_numpy()

# Get new class:
newSampleClass = svmModel.predict(newSampleVectorized)
print(" According to the SVM, the class is: " + str(newSampleClass[0]) + ": " + classesDict[newSampleClass[0]])

if runGridSearch:

    print("[INFO] --- Running Grid Search...")
    # Hyperparameter optimization using Grid Search:
    parameters = {"kernel": ("linear", "rbf"), "C": (2, 4, 8, 16)}
    svm = svm.SVC()
    optimizedSVM = GridSearchCV(svm, parameters, cv=5, n_jobs=parallelJobs)
    optimizedSVM.fit(trainingFeatures, trainingLabels)

    # Print hyperparameters & accuracy:
    print(optimizedSVM.best_params_)
    if crossValidateModel:
        svmAccuracy = cross_val_score(estimator=svmModel, X=trainingFeatures, y=trainingLabels, cv=cvFolds,
                                      n_jobs=parallelJobs, verbose=3)
        # Accuracy for each fold:
        print(svmAccuracy)

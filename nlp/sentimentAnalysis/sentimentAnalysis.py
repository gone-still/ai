# File        :   sentimentAnalysis.py
# Version     :   1.0.0
# Description :   Brief example of sentiment analysis using bag of words
#				  and a SVM.
               
# Date:       :   Mar 30, 2023
# Author      :   Ricardo Acevedo-Avila
# License     :   Creative Commons CC0

import nltk, random
from nltk.corpus import movie_reviews
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

# Download the movie_reviews data set:
nltk.download("movie_reviews")

# Check out some dataset info:
print("Samples:", len(movie_reviews.fileids()))
print("Classes:", movie_reviews.categories())
print("Words: ", movie_reviews.words()[:100])
print("Files: ", movie_reviews.fileids()[:10])

# Data set is re-arranged to build a table where each row stores a tuple.
# the tuple has the sample text (document) and its class (pos/neg):
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Shuffle the dataset:
random.seed(123)
random.shuffle(documents)

# Get some descriptive statistics:
#   Corpus Size (Number of Documents/Samples)
#   Corpus Size (Number of Words)
#   Distribution of the two classes

print("Number of Reviews/Documents: {}".format(len(documents)))
print("Corpus Size (words): {}".format(np.sum([len(d) for (d, l) in documents])))
print("Sample Text of Doc 1:")

# First 50 words (including punctuation) of the first document
print(" ".join(documents[0][0][:10]) + "...")

# Check Sentiment Distribution of the Current Dataset
# Count the number of samples/documents classified as "positive"
# and "negative"
sentimentDistr = Counter([label for (words, label) in documents])
print("Class distribution (Total): ", sentimentDistr)

# Dataset division, training: 80% testing: 20%
train, test = train_test_split(documents, test_size=0.20, random_state=420)

# Class/Sentiment distribution for Train and Test:
trainDistribution = Counter([label for (words, label) in train])
testDistribution = Counter([label for (words, label) in test])
print("Class distribution (Train): ", trainDistribution)
print("Class distribution (Test): ", testDistribution)

# Split the training data set into features (the vectorized
# words) and labels (classes):

# Training:
# Each sample is the complete commentary, including spaces and punctuation:
trainSamples = [" ".join(words) for (words, label) in train]
trainLabels = [label for (words, label) in train]

# Testing
testSamples = [" ".join(words) for (words, label) in test]
testLabels = [label for (words, label) in test]

# Sample/Text vectorization:
# Use bag of words to vectorize each sample:
# Bag of words is applied via the TfidfVectorizer object,
# Each token is a word with the characters a-z or A-Z,
# min_df is the minimal threshold of word frequency needed to
# not filter out the word:
wordVectorizer = TfidfVectorizer(min_df=10, token_pattern=r'[a-zA-Z]+')

# Use fit_transform() on the training set and transform() on the testing set.
# Fit Train
trainSamplesVectorized = wordVectorizer.fit_transform(trainSamples)
# Fit Test
testSamplesVectorized = wordVectorizer.transform(testSamples)

# Check out the vectorized dataset. Vector size (second dimension)
# must match for both training and testing
print("Train Samples shape: ", trainSamplesVectorized.shape)
print("Test Samples shape: ", testSamplesVectorized.shape)

# Check out the word encoding per document/sample for
# the training data set:
wordEncoding = pd.DataFrame(trainSamplesVectorized.toarray(), columns=wordVectorizer.get_feature_names())

# Classification via SVM:
# Train the SVM
# SVM hyperparameters: C = 0.8 with linear kernel:
svmModel = svm.SVC(C=8.0, kernel="linear")
svmModel.fit(trainSamplesVectorized, trainLabels)

# Evaluate SVM using cross validation, use 5 folds:
cvFolds = 5
svmAccuracy = cross_val_score(estimator=svmModel, X=trainSamplesVectorized, y=trainLabels, cv=cvFolds, n_jobs=-1)
# Accuracy for each fold:
print(svmAccuracy)

# Test the SVM:
maxTestSamples = 10
testVector = testSamplesVectorized[:maxTestSamples]
svmPredictions = svmModel.predict(testVector)
print(svmModel.score(testSamplesVectorized, testLabels))

# Get confusion matrix and its plot:
confusionMatrix = confusion_matrix(testLabels[:maxTestSamples], svmPredictions, labels=svmModel.classes_,
                                   normalize="all")
disp = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=svmModel.classes_)
disp.plot()
plt.show()

# Classify two new samples/documents/reviews:
newReview = ["This book is crappy. The cover and content are both weird.",
             "This book is fucking great. I recommend it!"]
newReviewVectorized = wordVectorizer.transform(newReview)
newReviewClass = svmModel.predict(newReviewVectorized)
print(newReviewClass)

# Hyperparameter optimization using Grid Search:
parameters = {"kernel": ("linear", "rbf"), "C": (1, 4, 8, 16, 32)}
svm = svm.SVC()
optimizedSVM = GridSearchCV(svm, parameters, cv=10, n_jobs=-1)
optimizedSVM.fit(trainSamplesVectorized, trainLabels)

# Print hyperparameters & accuracy:
print(optimizedSVM.best_params_)
print(optimizedSVM.score(testSamplesVectorized, testLabels))

# Re-plot the confusion matrix:
svmPredictions = optimizedSVM.predict(testVector)
confusionMatrix = confusion_matrix(testLabels[:maxTestSamples], svmPredictions, labels=optimizedSVM.classes_,
                                   normalize="all")
disp = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=optimizedSVM.classes_)
disp.plot()
plt.show()

# Re-classify reviews:
newReviewClass = optimizedSVM.predict(newReviewVectorized)
print(newReviewClass)


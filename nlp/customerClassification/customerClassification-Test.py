# File        :   customerClassification-Test.py
# Version     :   1.0.0
# Description :   [Stand-Alone Test]
#                 An unpaid "challenge" from company X to apply NLP techniques
#                 to classify customer requests into products. The shitty dataset has
#                 already been pre-processed with Weka's CfsSubsetEval to drop the
#                 meaningless features and reduce dimensionality.

# Date:       :   Apr 07, 2023
# Author      :   Ricardo Acevedo-Avila
# License     :   Creative Commons CC0


import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


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


# Project Path:
projectPath = "D://dataSets//nlp-case//"

# File Names:
modelName = "customerClassifier-SVM.sav"
vocabularyName = "savedVocabulary.pkl"
dictionaryName = "classDict.pkl"
vectorizerName = "wordVectorizer.pkl"

# Classification via SVM:
# Load the model:
print("[INFO] --- Loading the SVM model...")
modelPath = projectPath + modelName
print(" Reading: " + modelPath)
svmModel = pickle.load(open(modelPath, "rb"))

# Load the word vectorizer:
print("[INFO] --- Loading the Word Vectorizer object...")
vectorizerPath = projectPath + vectorizerName
print(" Reading: " + vectorizerPath)
wordVectorizer = pickle.load(open(vectorizerPath, "rb"))

# Load the class dictionary:
print("[INFO] --- Loading the Class Dictionary file...")
dictionaryPath = projectPath + dictionaryName
print(" Reading: " + dictionaryPath)
classesDict = pickle.load(open(dictionaryPath, "rb"))

print("[INFO] --- The Classes Dictionary:")
for i, key in enumerate(classesDict):
    # Get current class:
    currentClass = classesDict[key]
    print(" " + str(key) + " --> " + str(currentClass))

# Classify a new document:
print("[INFO] --- Classifying New Sample")

newSample = ["A claim was opened on  XXXX  with  Paypal  for a transaction, #  XXXX ,  XXXX  from  XXXX  on  XXXX . "
             "I received a message today asking me to return the item at my own cost to  XXXX . The reason for dispute "
             "is the item was damage by a local   XXXX   store and they refused to refund the item. The item was taken "
             "to a local store to be monogrammed and the  rep ( s ) damaged the item by pressing initials too hard on "
             "the leather making the monogram too deep and damage the item. I attempted to explain this to several "
             "Paypal reps, howeve r, I was told the claim was closed out and it was not by myself. "
             "Papal reps refused to do anything further"]

# Vectorize new sample:
filteredWords = ["xxxx", " i ", "i ", " us ", " we ", " am ", "hello", "please", "bye", "company", "husband", "wife"]
newSample = wordFilter(newSample, filteredWords)
newSampleVectorized = wordVectorizer.transform(newSample)
newSampleVectorized = pd.DataFrame(newSampleVectorized.toarray(), columns=wordVectorizer.get_feature_names_out())

# Dataframe to numpy array:
newSampleVectorized = newSampleVectorized.to_numpy()

# Get new class:
newSampleClass = svmModel.predict(newSampleVectorized)
print(" According to the SVM, the class is: " + str(newSampleClass[0]) + ": " + classesDict[newSampleClass[0]])

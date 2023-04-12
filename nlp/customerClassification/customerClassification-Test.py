# File        :   customerClassification-Test.py
# Version     :   1.1.0
# Description :   [Stand-Alone Test]
#                 An unpaid "challenge" from company X to apply NLP techniques
#                 to classify customer requests into products. The shitty dataset has
#                 already been pre-processed with Weka's CfsSubsetEval to drop the
#                 meaningless features and reduce dimensionality.

# Date:       :   Apr 12, 2023
# Author      :   Ricardo Acevedo-Avila
# License     :   Creative Commons CC0


import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer

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


# Project Path:
projectPath = "D://dataSets//nlp-case//"

downloadStopword = False

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

newSample = [
    "  I had this credit card that was closed and paid as agreed in XX/XX/XXXX. Citibank. In XX/XX/XXXX, they sued me "
    "after reopening without my knowledge and doubles the amount owed with fees. When I called them they say they are "
    "not the one suing me, the card was sold to / Assignments rights given to Citi XXXX in XXXX and no longer owned"
    " by them. They could not even find it in their system. I ended up speaking to a XXXX in their recorders section "
    "in NY, that told me all lawsuit records are processed through her, and they Citibank is not the Plaintiff where "
    "my cases is concerned. When they, and also I did call Citi XXXX, we got the same recording that the account is "
    "handled by a outside vendor, etc .... the 3rd party vendor is the attorney firm that bought it. They have been "
    "suing me since XXXX, and this is causing XXXX and financial hardship on me an my family. My XXXX is extremely "
    "high, my doctor can even XXXX it, because I 'm constantly under this XXXX.I have paid over {$4000.00} so far, "
    "and still owe the attorney over {$4000.00}. XXXX have paid them too, just about the same amount of money and "
    "I really do n't know what my attorney office is doing. They keep passing the case around the office to their"
    " different attorneys and redoing all request afresh as a new case. They have all the evidence they the attorney "
    "firm suing is fraudlently using citibank from day one, and will do nothing about it. Now after 8 years of suing, "
    "and about 10 years since the card is closed, I was just told by my firm we are going to court for a trial and "
    "they, my attorney firm are withdrawing. After over {$10000.00} in fees and proof they are fraudulent, they want "
    "to withdraw? HELP, HELP."]

# Vectorize new sample:
if downloadStopword:
    # Download the stop words list:
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

newSample = wordFilter(newSample, tokenizer, stopwordList)
newSampleVectorized = wordVectorizer.transform(newSample)
newSampleVectorized = pd.DataFrame(newSampleVectorized.toarray(), columns=wordVectorizer.get_feature_names_out())

# Dataframe to numpy array:
newSampleVectorized = newSampleVectorized.to_numpy()

# Get new class:
newSampleClass = svmModel.predict(newSampleVectorized)
print(" According to the SVM, the class is: " + str(newSampleClass[0]) + ": " + classesDict[newSampleClass[0]])

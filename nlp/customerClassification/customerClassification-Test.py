# File        :   customerClassification-Test.py
# Version     :   1.0.1
# Description :   [Stand-Alone Test]
#                 An unpaid "challenge" from company X to apply NLP techniques
#                 to classify customer requests into products. The shitty dataset has
#                 already been pre-processed with Weka's CfsSubsetEval to drop the
#                 meaningless features and reduce dimensionality.

# Date:       :   Apr 08, 2023
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

        # Apply filter word. Replaces targets with empty string:
        # for f in range(len(filteredWords)):
        #    targetWord = filteredWords[f]
        #    currentMessage = currentMessage.replace(targetWord, " ")

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

newSample = ["My mortgage was sold in XX/XX/XXXX from XXXX to Specialized Loan Servicing LLC ( SLS ). I contacted SLS "
             "regarding an application for a loan modification due to some unforeseen circumstances that had happened "
             "regarding my home. They mailed me out an application and I also printed one from their online website. I "
             "contacted them regarding how I should return the application. The lady said I could mail it to the "
             "address on the application, email it to the address on the application, or fax it to the number on the "
             "application. Since I had pictures to attach to the application I decided it would be best just to send "
             "it via email. The email address they gave was XXXXXXXXXXXX. Every time I sent an email I got a "
             "confirmation email back saying my documents were received and waiting to be processed. Starting on "
             "XX/XX/XXXX, I started receiving calls from SLS 's main number XXXX the caller identified herself on my "
             "voice mail as XXXX XXXX with the underwriting department at Specialized Loan Servicing. After a few days"
             " of phone tag, we were finally able to touch base. I spoke with XXXX on XX/XX/XXXX and she was wanting to "
             "discuss my modification approval for a HAMP modification. She also said the paperwork had been mailed out "
             "but since I had n't gotten it she would email it to me and that I would n't be able to open the email "
             "without a code and the code would be my social security number. She said this was for my protection. "
             "I received the paperwork and looked it over. The loan number on the paperwork was mine, the amounts "
             "added up, the phone numbers were to SLS so everything seemed legit especially since the only people who "
             "have my loan number were me and SLS. Remember loan numbers are NOT public information. I signed the "
             "document and sent it back to SLS 's email at XXXXXXXXXXXX and got a confirmation email back saying it was "
             "received and being processed. I also made 2 payments towards my trial period payments totaling {$1200.00}. "
             "On XX/XX/XXXX, I received a call from SLS touching base to see how things were going and to let me know "
             "they were missing some documents to finish up a loan modification. I told him I always spoke with XXXX, "
             "he said that XXXX worked out of their XXXX office and it was only by chance that I always got her. "
             "I explained I had already signed a modification agreement with them that XXXX had sent me and made 2 of "
             "my three trial period payments. He looked at the agreement ( 6 days after they received it ) and told me "
             "it was n't something they had sent me. Then tells me the numbers on the agreement were n't to SLS he said "
             "that the number for where my loan was at was XXXX. Obviously, XXXX did n't have very good XXXX because "
             "that is the exact number that was on my papers. I have made a report with my local sheriff 's office,"
             " the white collar crimes division of the FBI, and the fraud department with HAMP."]

# Vectorize new sample:

# Download the stop words list:
nltk.download("stopwords")
# Tokenization of text
tokenizer = ToktokTokenizer()
# Setting English stopwords
stopwordList = nltk.corpus.stopwords.words("english")
# Add some custom words:
stopwordList.append("xxxx")
stopwordList.append("company")

newSample = wordFilter(newSample, tokenizer, stopwordList)
newSampleVectorized = wordVectorizer.transform(newSample)
newSampleVectorized = pd.DataFrame(newSampleVectorized.toarray(), columns=wordVectorizer.get_feature_names_out())

# Dataframe to numpy array:
newSampleVectorized = newSampleVectorized.to_numpy()

# Get new class:
newSampleClass = svmModel.predict(newSampleVectorized)
print(" According to the SVM, the class is: " + str(newSampleClass[0]) + ": " + classesDict[newSampleClass[0]])

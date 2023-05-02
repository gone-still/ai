# File        :   bookRecommender.py
# Version     :   1.1.0
# Description :   Script that implements a book recommendation system
#                 based on: https://github.com/WillKoehrsen/wikipedia-data-science/blob/master/notebooks/
#                 Book%20Recommendation%20System.ipynb
# Date:       :   May 01, 2023
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT

import numpy as np
import random
import json

from itertools import chain
from collections import Counter, OrderedDict

from tensorflow.keras.models import load_model
from bookEmbeddings import booknet

import matplotlib.pyplot as plt


# Returns ordered dictionary of counts of objects in input list:
def count_items(inputList):
    # Create a counter object
    counts = Counter(inputList)

    # Sort by highest count first and place in ordered dictionary
    counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    counts = OrderedDict(counts)

    return counts


# Generates batches of samples for training
# The code below creates a generator that yields batches of samples each time it is called. Neural networks are trained
# incrementally - a batch at a time - which means that a generator is a useful function for returning examples on which
# to train. Using a generator alleviates the need to store all of the training data in memory which might be an issue if
# we were working with a larger dataset such as images.
def generate_batch(pairs, n_positive=50, negative_ratio=1.0, classification=False):
    # Compute the batch size and the positive and negative pairs ratios:
    batch_size = n_positive * (1 + negative_ratio)
    batch = np.zeros((batch_size, 3))

    # Get the unique entries in the positive pair list:
    pairs_set = set(pairs)

    # Adjust label based on task
    if classification:
        neg_label = 0
    else:
        neg_label = -1

    # This creates a generator, called by the neural network during
    # training...
    while True:
        # Randomly choose positive examples
        for idx, (book_id, link_id) in enumerate(random.sample(pairs, n_positive)):
            batch[idx, :] = (book_id, link_id, 1)

        # Increment idx by 1
        idx += 1

        # Add negative examples until reach batch size
        while idx < batch_size:

            # random selection
            random_book = random.randrange(len(books))
            random_link = random.randrange(len(links))

            # Check to make sure this is not a positive example
            if (random_book, random_link) not in pairs_set:
                # Add to batch and increment index
                batch[idx, :] = (random_book, random_link, neg_label)
                idx += 1

        # Make sure to shuffle order
        np.random.shuffle(batch)

        # The yield statement suspends a function’s execution and sends a value back to the caller, but retains
        # enough state to enable the function to resume where it left off. When the function resumes, it continues
        # execution immediately after the last yield run. This allows its code to produce a series of values over time,
        # rather than computing them at once and sending them back like a list.
        yield {'book': batch[:, 0], 'link': batch[:, 1]}, batch[:, 2]


# The function below takes in either a book or a link, a set of embeddings, and returns the n most similar items
# to the query. It does this by computing the dot product between the query and embeddings.:

def find_similar(name, weights, target_index=None, index_target=None, index_name="book", n=10, least=False):
    # Select index and reverse index:
    index = target_index
    rindex = index_target

    # Check to make sure "name" is in index
    try:
        # Calculate dot product between book and all others
        dists = np.dot(weights, weights[index[name]])
    except KeyError:
        print("Query Not Found.")
        return None

    # Sort distance indexes from smallest to largest
    sorted_indices = np.argsort(dists)
    # Reverse the array so most similar book indices are first -> Most similar book to least similar book
    sorted_indices = sorted_indices[::-1]

    # If specified, find the least similar:
    if least:
        # Take the last n from sorted distances
        sorted_indices = sorted_indices[-n:]
        # Reverse so least similar is first:
        sorted_indices = sorted_indices[::-1]
        print(f"{index_name.capitalize()}s furthest from {name}.\n")

    # Otherwise find the most similar:
    else:
        # Take the first n sorted distances:
        sorted_indices = sorted_indices[:n]
        print(f"{index_name.capitalize()}s closest to {name}.\n")

    # Store the most/least similar book titles and distances here:
    bookList = []
    for c in sorted_indices:
        # Get book title & score:
        bookTitle = rindex[c]
        bookScore = dists[c]
        # Store them in the return list:
        bookList.append((bookTitle, bookScore))

    return bookList


# Project Path:
projectPath = "D://dataSets//books//"
datasetFilename = "found_books_filtered.ndjson"
modelFilename = "bookRecommender.model"

# DNN options:
loadModel = True
saveModel = False

# The DNN can model the problem as regression
# or classification, if False, the problem is
# classification:
reggressionMode = False
activationLayer = "softmax"

trainingEpochs = 20

# Learning rate factor:
gamma = 2.0
learningRate = gamma * 1e-3

# Store the book dataset in this list.
# Each book contains the title, the information from the Infobox book template, the internal wikipedia links,
# the external links, the date of last edit, and the number of characters in the article
# (a rough estimate of the length of the article).
books = []

# Open file in read mode and load it to the list:
filename = projectPath + datasetFilename
with open(filename, "r") as fin:
    # Append each line to the books
    for l in fin:
        currentLine = json.loads(l)
        books.append(currentLine)

# Some of these entries are not books. They appear to be Wikipedia articles.
# Let's filter them out from the dataset and store them here:
books_with_wikipedia = []
# Filtered books go here:
filteredBooks = []
for i, book in enumerate(books):
    # Get book title (stored in the first list position):
    bookTitle = book[0]
    # Filter the ones that start with the string "Wikipedia:"
    if "Wikipedia:" in bookTitle:
        # Non-book:
        books_with_wikipedia.append(book)
    else:
        # Book:
        filteredBooks.append(book)

    # print(i, bookTitle)

# Check out the number of filtered books:
books = filteredBooks
totalBooks = len(books)
print("[INFO] -- Total Books:")
print(" Found: " + str(totalBooks) + " books")

# Check out the info contained in a sample book:
sample = 21
print("[INFO] -- Book Sample:")
print(" Fields: " + str(len(books[sample])))
print(" Info: " + str(books[sample]))

# We will only use the wikilinks, which are saved as the
# third element (index 2) for each book:
print(" --> Wikilinks/Keywords: " + str(books[sample][2]))

# Book mapping
# [Direct] Book title -> Book integer
# [Reverse] Book integer -> Book title
# First we want to create a mapping of book titles to integers. When we feed books into the
# embedding neural network, we will have to represent them as numbers, and this mapping will
# let us keep track of the books.
# We'll also create the reverse mapping, from integers back to the title.
# book_index = {book[0]: idx for idx, book in enumerate(books)}

book_index = {}
index_book = {}

for i, book in enumerate(books):
    # Get current book title:
    bookTitle = book[0]
    # Create title -> integer dictionary:
    book_index[bookTitle] = i
    # Create integer -> title dictionary:
    index_book[i] = bookTitle

# Check out the mapping:
print("[INFO] -- Book Mapping:")
print(" Book integer:", book_index["Anna Karenina"])
print(" Book title:", index_book[22494])
print(" Keywords:", books[22494][2])

# Wikilinks/keywords exploration:
# Create a flat list of all the lists of keywords. The chain method takes lists as arguments, therefore a * is used to
# unpack the list:
# flatList = chain(*nestedList)
# flatList  = list(flatList)
wikilinks = list(chain(*[book[2] for book in books]))

# Remove duplicates:
uniqueWikilinks = set(wikilinks)
# Count entries:
uniqueWikilinksCount = len(set(wikilinks))
# Print number of uniques:
print("[INFO] -- There are " + str(uniqueWikilinksCount) + " unique wikilinks.")

# Links to other books. The links appear as the book title itself:
wikilinks_other_books = []
for link in wikilinks:
    # Look for book title in the book_index list:
    if link in book_index.keys():
        wikilinks_other_books.append(link)

# Remove duplicates:
wikilinks_other_books = list(set(wikilinks_other_books))
# Count entries:
wikilinks2BooksCount = len(wikilinks_other_books)
print("[INFO] -- There are " + str(wikilinks2BooksCount) + " unique wikilinks to other books.")

# Most Linked-to Articles
# Which articles (other books) are most linked to by books on Wikipedia.

# For each book, get the wikilinks/keywords. Remove duplicates. Flatten the nested list
# and set it to the unique_wikilinks list:
unique_wikilinks = list(chain(*[set(book[2]) for book in books]))

# "Normalize" all keywords to lower case only, as to avoid inconsistencies
# with multiple cased-entries of the same keyword:
unique_wikilinks = [link.lower() for link in unique_wikilinks]

# Get the wikilinks with the highest count amongst books. That is, the most frequent keywords.
# The function returns them as a list of tuples (keyword, count):
wikilink_counts = count_items(unique_wikilinks)

# Check the top 10 keywords:
totalKeywords = 10

# Ordered dict to list:
wikilinks_list = list(wikilink_counts.items())
print("[INFO] -- There are " + str(len(wikilinks_list)) + " unique wikilinks after case normalization.")
print("[INFO] -- These are the Top " + str(totalKeywords) + " wikilinks/keywords:")

for i in range(totalKeywords):
    currentTuple = wikilinks_list[i]
    print(" ", i, currentTuple)

# Filter/Remove Most Popular Wikilinks:
# The most used keywords do not contribute much to the data. Remove the most-frequently used terms.
# The idea is similar to stop words in NLP:
stopWords = ["hardcover", "paperback", "hardback", "e-book", "wikipedia:wikiproject books",
             "wikipedia:wikiproject novels"]
for t in stopWords:
    # Remove from keyword list:
    unique_wikilinks.remove(t)
    # Remove from frequency dictionary:
    _ = wikilink_counts.pop(t)

# Final filtered keywords/wikilinks go here:
links = []
# Just use the >= 4 most relevant keywords:
minKeywordCount = 4
for dictTuple in wikilink_counts.items():
    # Get the keyword count:
    keywordCount = dictTuple[1]
    # Frequency-filter it:
    if keywordCount >= minKeywordCount:
        # Store the keyword:
        links.append(dictTuple[0])

print("[INFO] -- Total frequency-filtered keywords/wikilinks: " + str(len(links)))

# Most Linked-to Books
# Let's look at the books that are mentioned the most by other books on Wikipedia. We'll take the set of links
# for each book so that we don't have multiple counts for books that are linked to by another book more than once.

# Find set of book wikilinks for each book
unique_wikilinks_books = list(
    chain(*[list(set(link for link in book[2] if link in book_index.keys())) for book in books]))

# Count the number of books linked to by other books
wikilink_book_counts = count_items(unique_wikilinks_books)

# Check the top 10 books:
totalBooks = 10

# Ordered dict to list:
wikilink_book_counts = list(wikilink_book_counts.items())
print("[INFO] -- These are the Top " + str(totalKeywords) + " linked books:")

for i in range(totalBooks):
    currentTuple = wikilink_book_counts[i]
    print(" ", i, currentTuple)

# Keyword/wikilink mapping
# [Direct] Keyword/wikilink -> Keyword/wikilink integer
# [Reverse] Keyword/wikilink integer -> Keyword/wikilink

link_index = {}
index_link = {}

for i, link in enumerate(links):
    # Get current keyword:
    keyword = links[i]
    # Create title -> integer dictionary:
    link_index[keyword] = i
    # Create integer -> title dictionary:
    index_link[i] = keyword

# Check out the mapping:
print("[INFO] -- Keyword/wikilink Mapping:")
print(" There are: " + str(len(link_index)) + " wikilinks that will be used.")
print(" Link integer:", link_index["the economist"])
print(" Link title:", index_link[300])

# Build the Training Set
# We are going to treat this as a supervised learning problem: given a pair (book, link), we want the
# neural network to learn to predict whether this is a legitimate pair - present in the data - or not.
# The input (positive and negative ) pairs list will consist of tuples of every (book, link) pairing
# on all of Wikipedia.

pairs = []

# Iterate through each book
for book in books:
    # Iterate through each keyword/wikilink in each book:
    for link in book[2]:
        # to lowercase:
        currentLink = link.lower()
        # Check if link exists in the link_index dictionary:
        if currentLink in link_index:
            # Get its integer encoding:
            currentInteger = link_index[currentLink]
            # Into the pairs list:
            # Book title is stored in book[0]
            currentBook = book_index[book[0]]
            pairs.append((currentBook, currentInteger))

print("[INFO] -- Total Pairs / Total Links / Total Books:")
print(" ", len(pairs), len(links), len(books))
print(" Sample pair: ", pairs[5000])

# Check out some samples from the pairs list:
print("[INFO] -- Decoded Sample pairs:")
print(" ", index_book[pairs[5000][0]], index_link[pairs[5000][1]])
print(" ", index_book[pairs[900][0]], index_link[pairs[900][1]])

# Create the dataset with positive and negative pairs:
random.seed(100)

# n_positive is the number of positive pairs in the batch
# negative_ratio is the ration eg (double the positive pairs) of the
# negative pairs
# This bit generates 2 positive pairs and 4 negative pairs in a batch
# of 2 + 4 = 6 samples:
# x is a dictionary of book and keyword integers, y is an array of labels
x, y = next(generate_batch(pairs, n_positive=2, negative_ratio=2))

# Show a few example training pairs:
print("[INFO] -- Training dataset examples:")
for i, (label, b_idx, l_idx) in enumerate(zip(y, x["book"], x["link"])):
    bookTitle = index_book[b_idx]
    bookLink = index_link[l_idx]
    print(f" {i} Book: {bookTitle:30} Link: {bookLink:40} Label: {label}")

# Load or train model from scratch:
if loadModel:
    # Get model path + name:
    modelPath = projectPath + modelFilename
    print("[INFO] -- Loading DNN Model from: " + modelPath)
    # Load model:
    model = load_model(modelPath)
    model.summary()
else:
    print("[INFO] -- Creating + Fitting DNN Model from scratch")

    # Set up the DNN:
    model = booknet.build(embedding_size=50, book_embedding_input_len=len(book_index),
                          link_embedding_input_len=len(link_index), classification=reggressionMode,
                          activationFunction=activationLayer,
                          alpha=learningRate, epochs=trainingEpochs)
    model.summary()

    # Print some debugging info:
    if reggressionMode:
        dnnMode = "Regression"
    else:
        dnnMode = "Classification"
    print("[INFO] -- Mode: " + dnnMode + " Epochs: " + str(trainingEpochs) + " Learning Rate: " + str(
        learningRate) + " Activation: " + activationLayer)

    # Set the samples generator:
    n_positive = 1024
    gen = generate_batch(pairs, n_positive, negative_ratio=2, classification=reggressionMode)

    # Train the DNN:
    H = model.fit(
        gen,
        steps_per_epoch=len(pairs) // n_positive,
        epochs=trainingEpochs,
        verbose=1
    )

    if saveModel:
        modelPath = projectPath + modelFilename
        print("Saving model to: " + str(modelPath))
        model.save(modelPath)

    # Plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()

    # Get the historical data:
    N = np.arange(0, trainingEpochs)

    history_dict = H.history
    print(history_dict.keys())

    # Plot values:
    plt.plot(N, H.history["loss"], label="train_loss")
    # plt.plot(N, H.history["val_loss"], label="val_loss")
    # plt.plot(N, H.history["accuracy"], label="train_acc")
    # plt.plot(N, H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")

    # Save plot to disk:
    plt.savefig(projectPath + "graph.png")
    plt.show()

# Extract Embeddings and Analyze them
# Extract embeddings
book_layer = model.get_layer("book_embedding")
book_weights = book_layer.get_weights()[0]
print("[INFO] -- Book Embeddings Shape: ")
print(" ", book_weights.shape)

# Normalize embeddings:
# Get the L2 Norm of all the embeddings as a row, then transpose it:
books_norm = np.linalg.norm(book_weights, axis=1).reshape((-1, 1))
# Divide each embedding by the square root of the sum of squared components (its L2 Norm):
book_weights = book_weights / books_norm
# print(book_weights[0][:10])
# print(np.sum(np.square(book_weights[0])))

# Find Similar Books, the function returns a list of tuples containing the n most/least similar
# books, in the format (book title, book similarity):
targetBook = "Trainspotting (novel)"
totalSimilarBooks = 10
leastSimilar = False
similarBooks = find_similar(targetBook, book_weights, book_index, index_book, "book", totalSimilarBooks, leastSimilar)

print("[INFO] -- Similar Books to: " + targetBook)

# Make sure the target book was found in the dataset!
if similarBooks:
    # Check out the results:
    for i in range(len(similarBooks)):
        # Get book title:
        bookTitle = similarBooks[i][0]
        # Get book score/distance/similarity:
        bookScore = similarBooks[i][1]

        # Print the info:
        print(f" {i} Book: {bookTitle:50} Similarity: {bookScore:.{2}}")

else:
    # Book not found:
    print(" Book: " + targetBook + " not in dataset.")

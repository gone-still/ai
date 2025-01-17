# File        :   movieRecommender.py
# Version     :   1.0.0
# Description :   Script that implements a movie recommendation system based on genre keywords

# Date:       :   May 19, 2023
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT


import pandas as pd
import numpy as np
import random
import tensorflow as tf
from movieNet import movieNet
from tensorflow.keras.models import load_model

from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt


# Returns True if an array is found in an array list:
def isArrayInList(inputArray, arrayList):
    return next((True for item in arrayList if item is inputArray), False)


# Batch Generator:
def generateBatch(pairs, moviesDicts, genresDicts, moviesGenresDict, n_positive=50, negative_ratio=1.0,
                  classification=False, startVal=0, pickRandom=False):
    # Get dictionaries:
    forwardMovieDict = moviesDicts[0]
    reverseMovieDict = moviesDicts[1]
    totalMovies = len(forwardMovieDict)

    forwardGenresDict = genresDicts[0]
    reverseGenresDict = genresDicts[1]
    totalGenres = len(forwardGenresDict)

    # Compute the batch size and the positive and negative pairs ratios:
    batch_size = n_positive * (1 + negative_ratio)

    # Get vector length from one sample:
    genresVectorLength = pairs[0].shape[0] - 1  # subtract the movie id
    batch = np.zeros((batch_size, genresVectorLength + 2))

    # Adjust label based on task
    if classification:
        negLabel = 0
    else:
        negLabel = -1

    # Set list index for sequential
    # sampling:
    listStart = startVal

    # This creates a generator, called by the neural network during
    # training...
    while True:
        # Check if random ore sequential samples from the dataset should
        # be drawn:
        if pickRandom:
            # Randomly choose n positive examples from the pairs list:
            randomSamples = random.sample(pairs, n_positive)

        else:
            # Sequential sampling from dataset:
            # Watch the sampling index:
            if listStart >= len(pairs):
                # On every "Epoch", shuffle the dataset
                random.shuffle(pairs)
                # Sampling index is reset:
                listStart = 0

            # Compute end of batch index:
            listEnd = listStart + n_positive
            # Sequentially slice the batch from the dataset:
            randomSamples = pairs[listStart:listEnd]

        # Store the positive random samples in the batch array:
        for i in range(len(randomSamples)):
            # Get current sample numpy array:
            currentSample = randomSamples[i]
            # Set movie id + genre vector:
            batch[i][0:-1] = currentSample
            # Set (positive) class:
            batch[i][-1] = 1

        # Increment i by 1
        i += 1

        # Add negative examples until reach batch size
        while i < batch_size:

            # Random selection of movie and keywords vector:
            randomMovie = random.randrange(totalMovies)  # 100 total movies, randomly choose from 0 to 99
            movieTitle = reverseMovieDict[randomMovie]  # Just checking the actual movie title...

            # Get genres for this movie:
            randomMovieGenres = moviesGenresDict[randomMovie]
            randomGenresVector = np.zeros(genresVectorLength)

            # Get random genres for this movie excluding those in list:
            for j in range(genresVectorLength):
                # Get random genre from 0 to 19 excluding the real genres:
                randomGenre = random.choice(list(set(range(0, totalGenres + 1)) - set(randomMovieGenres)))
                # Into the random vector of genres:
                randomGenresVector[j] = randomGenre
                # Exclude this genre for this sample. Genres cannot repeat more than once in the same sample:
                randomMovieGenres = np.append(randomMovieGenres, randomGenre)

            # Build random vector, insert movie id at position 0 of a
            # larger numpy array made of the random movie values:
            randomVector = np.insert(randomGenresVector, 0, randomMovie)

            # Check to make sure this is not a positive example
            isInList = isArrayInList(randomVector, pairs)
            if not isInList:
                # Add to batch and increment index
                batch[i][0:genresVectorLength + 1] = randomVector
                batch[i][-1] = negLabel
                i += 1

        # Make sure to shuffle order
        np.random.shuffle(batch)
        # print(" Generated a batch with: "+str(len(batch))+" samples")

        # Prepare sequential dataset slice index for next iteration:
        listStart = listStart + n_positive

        # The yield statement suspends a function’s execution and sends a value back to the caller, but retains
        # enough state to enable the function to resume where it left off. When the function resumes, it continues
        # execution immediately after the last yield run. This allows its code to produce a series of values over time,
        # rather than computing them at once and sending them back like a list.
        outDict = {"movie": batch[:, 0], "genres": batch[:, 1:genresVectorLength + 1]}, batch[:, -1]

        yield outDict


# Normalizes embeddings:
def normalizeEmbeddings(inputArray):
    # Normalize embeddings:
    # Get the L2 Norm of all the embeddings as a row, then transpose it:
    norm = np.linalg.norm(inputArray, axis=1).reshape((-1, 1))
    # Divide each embedding by the square root of the sum of squared components (its L2 Norm):
    normalizedVectors = inputArray / norm
    # print(np.sum(np.square(normalizedVectors[0])))
    return normalizedVectors


# Check if GPU utilization is possible:
print(tf.config.list_physical_devices("GPU"))

# Project Path:
projectPath = "D://dataSets//movies//"

# File Names:
datasetName = "imdb_movie_keyword.csv"
modelFilename = "movieRecommender.model"

# Batch generation options:
randomRows = True

# DNN Setup:
embeddingSize = (100, 100)
classificationMode = False
activationLayer = "softmax"
trainingEpochs = 10
gamma = 10.0 * 2.0
learningRate = gamma * 1e-2

# DNN options:
saveModel = False
loadModel = True

# Read the CSV Dataset:
inputDataset = pd.read_csv(projectPath + datasetName)

# Check out the first 5 samples:
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

print("[INFO] --- Dataset 5 First Samples:")
print(inputDataset.head())

# Extract the genre column:
movieGenres = inputDataset["genre"]
movieGenres = movieGenres.to_frame()

# Create an extra column with a list of strings:
movieGenres["genres_list"] = movieGenres["genre"].str.split(",")
print(movieGenres.head())

# Split genres list into separate columns:
movieGenresSplit = pd.DataFrame(movieGenres["genres_list"].tolist()).fillna("").add_prefix("genre_")
movieGenres = pd.concat([movieGenres, movieGenresSplit], axis=1)

# Get number of columns used to split the genres:
(rows, columns) = movieGenresSplit.shape
print(" Movies Genres shape: ", (rows, columns))

# Add the new columns that will store the encoded
# genres
for i in range(columns):
    columnHeader = "encodedGenre_" + str(i)
    movieGenres[columnHeader] = 0

# Genres encoding. Create a dictionary with all
# the genres encoded into numerical values:
genresTempList = []

# Loop through the rows:
for i, row in enumerate(movieGenres.itertuples()):
    # Extract genre string:
    currentGenre = row.genre
    print(currentGenre)
    # Separate into individual strings:
    keywordsList = currentGenre.split(", ")

    # Encode strings:
    for j, k in enumerate(keywordsList):
        if k not in genresTempList:
            # Store in tempList:
            genresTempList.append(k)

totalGenres = len(genresTempList)
print(" Total genres: ", totalGenres)

# Set random seed:
random.seed(42)

# Shuffle list:
# random.shuffle(genresTempList)

# Create dictionaries from genre list:
genresDictionary = {}
genresDictionaryReverse = {}
keywordCount = 1

for g in range(totalGenres):
    # Get current genre:
    currentGenre = genresTempList[g]
    # Store in direct dictionary:
    genresDictionary[currentGenre] = keywordCount
    # Store in reverse dictionary:
    genresDictionaryReverse[keywordCount] = currentGenre
    keywordCount += 1

# Encode dataframe genre columns:
for i, row in enumerate(movieGenres.itertuples()):
    # Extract genre string:
    currentGenre = row.genre
    # print(currentGenre)
    # Separate into individual strings:
    keywordsList = currentGenre.split(", ")

    # Process genre strings:
    for j, k in enumerate(keywordsList):
        # Populate new encoded columns:
        genreCode = genresDictionary[k]
        # Into encoded column:
        columnHeader = "encodedGenre_" + str(j)
        movieGenres.loc[:, columnHeader][i] = genreCode

# Check out the first 5 samples of movie genres:
print("[INFO] -- Movie Genres Encoding:")
print(movieGenres.head())

# Extract the movie title:
movieTitles = inputDataset["movie_title"]

# Encode the categorical movie titles to numerical values,
# Explicit conversion for movie titles to categorical values:
movieTitlesEncoded = movieTitles.astype("category")
# Perform label encoding on the categorical values:
movieTitlesEncoded = movieTitlesEncoded.cat.codes
# Convert series to dataframe:
movieTitlesEncoded = movieTitlesEncoded.to_frame()
# Set column:
movieTitlesEncoded = movieTitlesEncoded.set_axis(["movie_title_encoded"], axis=1)
# Concatenate dataframe to original movie titles:
movieTitles = pd.concat([movieTitles, movieTitlesEncoded], axis=1)

# Check out the first 5 samples of movie titles and their encoding:
print("[INFO] -- Movie Titles Encoded:")
print(movieTitles.head())

# Create the movie_title -> code dictionary:
moviesDictionary = {}
# Create the dictionary mapping the movie title codes with their actual categorical titles:
moviesDictionary = dict(zip(movieTitles.movie_title, movieTitles.movie_title_encoded))

# Crete the complete (encoded) dataset:
columnsList = [movieTitlesEncoded]
for i in range(columns):
    columnHeader = "encodedGenre_" + str(i)
    columnsList.append(movieGenres[columnHeader])

encodedDataset = pd.concat(columnsList, axis=1)
print("[INFO] -- Movie Titles and Genres Encoded:")
print(encodedDataset.head())

# Get the dictionary lengths:
moviesDictionaryLength = len(moviesDictionary)
genresDictionaryLength = len(genresDictionary)

print(" Dictionary lengths:", (moviesDictionaryLength, genresDictionaryLength))

# Create movie (encoded) --> genres (encoded) dictionary:
moviesGenresDictionary = {}
moviesGenresDictionary = encodedDataset.set_index("movie_title_encoded").T.to_dict("list")

# Create reverse movie dictionary (Movie ID -> Movie Title):
moviesDictionaryReverse = {}
for movie in moviesDictionary:
    # Get value:
    moviId = moviesDictionary[movie]
    moviesDictionaryReverse[moviId] = movie

# Create pairs of real movie title + real (un ordered) genres:
pairs = []
samplesPerClass = 2
totalPairs = encodedDataset.shape[0]

# Sequentially sample a row from the encoded dataset:
for i in range(totalPairs):
    # Get sequential row:
    currentRow = encodedDataset.iloc[[i]]

    # Process row:
    rowList = currentRow.values[0]
    # Get movie code
    movieCode = rowList[0]

    # Get genre list:
    genreList = rowList[1:]
    totalGenres = genreList.shape[0]

    # For this movie, create a couple of samples with its
    # genres randomly shuffled:
    for g in range(samplesPerClass):
        # Create out numpy array:
        genresOut = np.zeros(totalGenres + 1)

        # Shuffle genres:
        np.random.shuffle(genreList)

        # Store in temp array:
        genresOut[0] = movieCode
        genresOut[1:] = genreList

        # Store shuffled genres into pair list:
        pairs.append(genresOut)

print("[INFO] -- Pairs dataset length:")
print("", len(pairs))

# Shuffle list:
random.shuffle(pairs)
# ... and again:
random.shuffle(pairs)

# Generate batch:
x, y = next(
    generateBatch(pairs=pairs, moviesDicts=(moviesDictionary, moviesDictionaryReverse),
                  genresDicts=(genresDictionary, genresDictionaryReverse), moviesGenresDict=moviesGenresDictionary,
                  n_positive=2, negative_ratio=2, classification=classificationMode, startVal=0, pickRandom=randomRows))

# Check batch info:
for i, (label, movieIndex, genresVector) in enumerate(zip(y, x["movie"], x["genres"])):
    movieTitle = moviesDictionaryReverse[int(movieIndex)]
    # Check genres vector:
    genresList = []
    # Convert genres id to actual strings:
    for genreId in genresVector:
        # 0 was used to identify empty cells:
        if genreId > 0:
            # Get it from dictionary
            currentGenre = genresDictionaryReverse[genreId]
        else:
            # Is not available/was empty cell:
            currentGenre = "N/A"
        # The list of actual genres as keywords:
        genresList.append(currentGenre)

    # Check the info:
    print("", i, movieTitle, genresList, label)

# Load or train model from scratch:
if loadModel:
    # Get model path + name:
    modelPath = projectPath + modelFilename
    print("[INFO] -- Loading DNN Model from: " + modelPath)
    # Load model:
    model = load_model(modelPath)
    # Get summary:
    model.summary()
else:
    print("[INFO] -- Creating + Fitting DNN Model from scratch:")

    # movies vocabulary length -> 100 [0, 99 (inclusive)] movies
    # genres vocabulary length -> 20 [0, 19 (inclusive)] genres
    model = movieNet.build(embeddingSize=embeddingSize, movieDictionaryLength=len(moviesDictionary),
                           genresDictionaryLength=len(genresDictionary) + 1, classification=classificationMode,
                           activationFunction=activationLayer, alpha=learningRate, epochs=trainingEpochs)
    # Get summary:
    model.summary()

    # Plot DNN model:
    graphPath = projectPath + "model_plot.png"
    plot_model(model, to_file=graphPath, show_shapes=True, show_layer_names=True)
    print("[INFO] -- Model graph saved to: " + graphPath)

    # Set the samples' generator:
    nPositive = 1024
    gen = generateBatch(pairs=pairs, moviesDicts=(moviesDictionary, moviesDictionaryReverse),
                        genresDicts=(genresDictionary, genresDictionaryReverse),
                        moviesGenresDict=moviesGenresDictionary,
                        n_positive=nPositive, negative_ratio=2, classification=classificationMode, startVal=0,
                        pickRandom=randomRows)

    # Train the DNN:
    H = model.fit(
        gen,
        steps_per_epoch=len(pairs) // nPositive,
        epochs=trainingEpochs,
        verbose=1
    )

    # Check if model needs to be saved:
    if saveModel:
        # Set model path:
        modelPath = projectPath + modelFilename
        print("[INFO] -- Saving model to: " + str(modelPath))
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
    plotPath = projectPath + "lossGraph.png"
    print("[INFO] -- Saving model loss plot to:" + plotPath)
    plt.savefig(plotPath)
    plt.show()

# Extract Genre embeddings:
genresLayer = model.get_layer("genresEmbedding")
genresWeights = genresLayer.get_weights()[0]
print("[INFO] -- Genre Embeddings Shape: ")
print(" ", genresWeights.shape)

# Extract Movie title embeddings:
moviesLayer = model.get_layer("movieEmbedding")
moviesWeights = moviesLayer.get_weights()[0]
print("[INFO] -- Movies Embeddings Shape: ")
print(" ", moviesWeights.shape)

# Normalize embeddings:
moviesWeightsNormalized = normalizeEmbeddings(moviesWeights)

# Get genres embedding from a genres query:
genresQuery = ["Thriller", "Horror"]
# Get the total genres used:
totalQueryKeywords = columns

# Store the total sum of each keyword embedding here:
genreEmbeddingsSum = 0

# For each keyword/genre in the list, get its embedding:
for g in range(totalQueryKeywords):
    # Get genre:
    if g < len(genresQuery):
        currentGenre = genresQuery[g]
        # Get genre encoding:
        genreCode = genresDictionary[currentGenre]
    else:
        # No genre is present:
        genreCode = 0

    # Get embedding:
    genreWeight = genresWeights[genreCode]
    # Accumulate-sum the embedding:
    genreEmbeddingsSum = genreEmbeddingsSum + genreWeight

# To numpy array:
genreEmbeddingsVector = np.array(genreEmbeddingsSum)

# Normalize the accumulated vector, this is the vector that will be measured
# with the movie title vector. Both are normalized now:
genresEmbeddingNorm = np.linalg.norm(genreEmbeddingsVector)
genreEmbeddingsVectorNormalized = genreEmbeddingsVector / genresEmbeddingNorm

# Get embeddings shape:
totalMovies = moviesWeights.shape[0]

# Compute dot product between the sum embeddings from the keywords/genres
# and the movies embeddings (this could be vectorized)...
# Store distances here:
embeddingDistances = []
for m in range(totalMovies):
    # Get current movie embedding:
    currentMovie = moviesWeights[m]
    currentMovieNormalized = moviesWeightsNormalized[m]

    # Dot product:
    currentDist = np.dot(currentMovieNormalized, genreEmbeddingsVectorNormalized)
    # a = currentMovie.reshape(1, 1, 60)
    # b = genreEmbeddingsVector.reshape(1, 1, 60)
    # kerasDot = Dot(normalize=True, axes=2)([a, b])
    # print(kerasDot)
    # s = kerasDot.numpy()
    # print(s)
    # Add to temp list:
    # embeddingDistances.append((currentDist, s))
    embeddingDistances.append(currentDist)

# Store the results here. They will be stored as
# the tuple: (distance, movie title, keywords/genres)
resultsList = []

# For all the dot-product produced distances:
for d in range(len(embeddingDistances)):

    # Get distance and decode movie title and genres:
    currentDistance = embeddingDistances[d]
    # currentDistance = embeddingDistances[d][0]
    # kerasDot = embeddingDistances[d][1]

    movieTitle = moviesDictionaryReverse[d]
    movieGenres = moviesGenresDictionary[d]

    # Concatenate all keywords in one, final string:
    genreString = ""

    # Create the results. Add every decoded keyword
    # to the final keyword string:
    for i in movieGenres:
        # Skip the keyword/genre = 0
        if i > 0:
            genreString = genreString + genresDictionaryReverse[i] + ", "

    # Drop the last ", ":
    genreString = genreString[:-2]

    # Into the results list:
    # resultsList.append((currentDistance, kerasDot[0][0][0], movieTitle, genreString))
    resultsList.append((currentDistance, movieTitle, genreString))

# Sort the list from largest to smallest distance:
resultsList.sort(reverse=True)
print("[INFO] -- Movie recommendations for: ", genresQuery)
# Print the recommendations:
for d in range(len(embeddingDistances)):
    print("", d, "-", resultsList[d])

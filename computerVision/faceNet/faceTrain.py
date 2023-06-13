import cv2
import os
from imutils import paths
from glob import glob
import numpy as np
import tensorflow as tf

import random


# Defines a re-sizable image window:
def showImage(imageName, inputImage):
    cv2.namedWindow(imageName, cv2.WINDOW_NORMAL)
    cv2.imshow(imageName, inputImage)
    cv2.waitKey(0)


# Writes an PGN image:
def writeImage(imagePath, inputImage):
    # imagePath = imagePath + ".png"
    cv2.imwrite(imagePath, inputImage, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print("Wrote Image: " + imagePath)


# Batch generator of positive & negatives pairs:
def generateBatch(pairs, n_positive=2, negative_ratio=2):
    # Get total number of pairs:
    totalPairs = len(pairs)

    # Compute the batch size and the positive and negative pairs ratios:
    batchSize = n_positive * (1 + negative_ratio)

    # The list of images (very row is a pair):
    batchSamples = []
    # The numpy array of labels (positive=1, negative=-1)
    batchLabels = np.zeros((batchSize, 1))

    # This creates a generator, called by the neural network during
    # training...
    while True:

        # Randomly choose n positive examples from the pairs list:
        choicesArray = np.arange(0, totalPairs, 1, dtype=int)
        positiveSamples = np.random.choice(choicesArray, n_positive, replace=True)

        totalPositiveSamples = len(positiveSamples)

        # Store the positive random samples in the batch array:
        for i in range(totalPositiveSamples):
            # Get current pair of images:
            randomSample = positiveSamples[i]
            currentSample = pairs[randomSample]

            # Check images (if proper data type):
            showImage("[Positive] Sample 1", currentSample[0])
            showImage("[Positive] Sample 2", currentSample[1])

            # Into the batch:
            batchSamples.append([currentSample[0], currentSample[1]])
            # Pair label:
            batchLabels[i] = 1

        # Set the sample index:
        sampleIndex = len(batchSamples)

        # Add negative examples until reach batch size
        while sampleIndex < batchSize:

            # Randomly generated negative sample row here:
            tempList = []
            pastClass = -1

            # Randomly choose two sample rows:
            for s in range(2):
                getRows = True
                while getRows:
                    # Get a random row from the positive pairs list,
                    # "Flatten" to an integer, since the return value is
                    # an array:
                    randomChoice = np.random.choice(choicesArray, 1)[0]

                    # Actually Get the random row:
                    randomRow = pairs[randomChoice]

                    # Get the sample's class:
                    rowClass = randomRow[-1]

                    if rowClass != pastClass:
                        # Randomly choose one of the two images:
                        randomChoice = random.randint(0, 1)
                        randomSample = randomRow[randomChoice]

                        showImage("randomSample: " + str(s), randomSample)

                        # Into the temp list:
                        tempList.append(randomSample)
                        # Stop the loop:
                        getRows = False

                        # Present is now past:
                        pastClass = rowClass

            # Got the two negative samples, thus the negative pair is ready:
            batchSamples.append(tempList)
            # This is a negative pair:
            batchLabels[sampleIndex] = -1
            # Increase the "batch processing" index:
            sampleIndex += 1

        # Make sure to shuffle list of samples and labels:
        # shuffleSamples(batchSamples, batchLabels)
        # random.shuffle(batchSamples)

        # python list To numpy array of numpy arrays...
        batchSamples = np.array(batchSamples)

        image1Arrays = batchSamples[:, 0:1]
        image2Arrays = batchSamples[:, 1:2]

        for h in range(6):
            print(h, batchLabels[h])
            showImage("[Batch] Sample 1", image1Arrays[h][0][0:32])
            showImage("[Batch] Sample 2", image2Arrays[h][0][0:32])

        outDict = {"image1": image1Arrays, "image2": image2Arrays}, batchLabels
        # outDict = {"image1": batch[:, 0], "image2": batch[:, 1:genresVectorLength + 1]}, batch[:, -1]
        yield outDict


# Set project paths:
projectPath = "D://dataSets//faces//"
datasetPath = projectPath + "out//cropped"

# Script Options:
randomSeed = 420

displayImages = False

imageSize = (32, 32)
embeddingSize = 50

samplesPerClass = 10

# Load each image path of the dataset:
print("[FaceNet Training] Loading images...")

# Get list of full subdirectories paths:
classesDirectories = glob(datasetPath + "//*//", recursive=True)
classesDirectories.sort()

# Trim root directory, leave only subdirectories names (these will be the classes):
rootLength = len(datasetPath)
classesImages = [dirName[rootLength + 1:-1] for dirName in classesDirectories]

# Create classes dictionary:
classesDictionary = {}
classCounter = 0
for c in classesImages:
    if c not in classesDictionary:
        classesDictionary[c] = classCounter
        classCounter += 1

print(classesDictionary)

# Get total classes:
totalClasses = len(classesImages)
print("Total Classes:", totalClasses, classesImages)

# Create the faces dataset as a dictionary:
facesDataset = {}

# Load images per class:
for c, currentDirectory in enumerate(classesDirectories):
    # Images for this class:
    imagePaths = list(paths.list_images(currentDirectory))
    totalImages = len(imagePaths)

    currentClass = classesImages[c]
    print("[FaceNet Training] Class: " + currentClass + " Samples: " + str(totalImages))

    # Create dictionary key:
    # Each key is a class name associated with
    # an array of samples for this class:
    facesDataset[currentClass] = []

    for currentPath in imagePaths:
        # Load the image:
        currentImage = cv2.imread(currentPath)

        # Pre-process the image for FaceNet input:
        currentImage = cv2.resize(currentImage, imageSize)
        # currentImage = currentImage.astype("float") / 255.0

        # Show the input image:
        if displayImages:
            showImage("Input image [Pre-processed]", currentImage)

        # Into the list of samples:
        facesDataset[currentClass].append(currentImage)

# Set random seed:
random.seed(randomSeed)
np.random.seed(randomSeed)

# Build the positive pairs dataset:
# Stores: (Class A - Sample 1, Class A - Sample 2, Class Code)
positivePairs = []

for currentClass in facesDataset:
    # Get total samples for this class:
    classSamples = len(facesDataset[currentClass])

    # processed samples counter:
    processedPairs = 0
    for i in range(samplesPerClass):

        choices = list(range(0, classSamples - 1))
        randomSamples = [0, 0]

        # Randomly choose a first sample:
        randomSamples[0] = random.choice(choices)

        # Randomly choose a second sample, excluding the one already
        # chosen:
        randomSamples[1] = random.choice(list(set(choices) - set([randomSamples[0]])))

        # Print the chosen samples:
        print("Processing pair: " + str(processedPairs), randomSamples)

        # Store sample pair here:
        tempList = []

        # Get images from dataset:
        for s in range(len(randomSamples)):
            sampleIndex = randomSamples[s]
            currentImage = facesDataset[currentClass][sampleIndex]
            # Show image
            if displayImages:
                showImage(currentClass + " Pair: " + str(s), currentImage)
            # Into the temp list:
            tempList.append(currentImage)

        # Finally, store class code:
        classCode = classesDictionary[currentClass]
        tempList.append(classCode)
        # Into the positive pairs list:
        positivePairs.append(tempList)
        processedPairs += 1

    # Done creating positive pairs for this class:
    totalPairs = len(positivePairs)
    print("Pairs Created: " + str(processedPairs) + ", Class: " + currentClass,
          "(" + str(classesDictionary[currentClass]) + ")", " Total: " + str(totalPairs))

# Convert uint type to float:
# positivePairs = np.array(positivePairs, dtype="float") / 255.0
# positivePairs = np.array(positivePairs)

# Shuffle the list of positive pairs:
# random.shuffle(positivePairs)

# Generate sample batch:
x, y = next(generateBatch(pairs=positivePairs, n_positive=2, negative_ratio=2))

# Check batch info:
for i, (label, img1, img2) in enumerate(zip(y, x["image1"], x["image2"])):
    # Check the info:
    print("Batch Sample: ", i, label)

    # Show the images:
    showImage("[Generator] Sample 1", img1[0][0:32])
    showImage("[Generator] Sample 2", img2[0][0:32])

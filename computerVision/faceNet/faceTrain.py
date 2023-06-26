# File        :   faceTrain.py
# Version     :   0.6.0
# Description :   faceNet training script

# Date:       :   Jun 25, 2023
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT

import cv2
import math

from imutils import paths
from glob import glob
import numpy as np
import random

from faceNet import faceNet

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

import matplotlib.pyplot as plt


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


# Receives two images and shows them in a window:
def showImages(windowName, imageList):
    # Horizontally concatenate images:
    outImage = cv2.hconcat(imageList)
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.imshow(windowName, outImage)
    cv2.waitKey(0)


# Shuffles a batch of lists (images) and a numpy array (labels) that
# share a "row" in a matrix:
def shuffleSamples(batchSamples, batchLabels):
    # Get total size (rows) of the batch/matrix:
    batchSize = batchLabels.shape[0]
    # Create the ascending array of choices:
    choicesArray = np.arange(0, batchSize, 1, dtype=int)
    # Shuffle the choices array:
    np.random.shuffle(choicesArray)

    # Output containers:
    shuffledList = []
    shuffledArray = np.zeros((batchSize, 1))

    # Shuffle the actual list and array:
    for i in range(batchSize):
        # Grab random choice:
        randomNumber = choicesArray[i]
        # Grab image pair from list and
        # label from array, imagePair
        # is also a list:
        shuffledList.append(batchSamples[randomNumber])
        shuffledArray[i] = batchLabels[randomNumber]

    return shuffledList, shuffledArray


# Batch generator of positive & negatives pairs:
def generateBatch(pairs, n_positive=2, negative_ratio=2):
    # Get total number of pairs:
    totalPairs = len(pairs)

    # Compute the batch size and the positive and negative pairs ratios:
    batchSize = n_positive * (1 + negative_ratio)

    # The numpy array of labels (positive=1, negative=-1)
    batchLabels = np.zeros((batchSize, 1))

    # This creates a generator, called by the neural network during
    # training...
    while True:

        # The list of images (very row is a pair):
        batchSamples = []

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
            if displayImages:
                showImage("[Positive] Sample 1", currentSample[0])
                showImage("[Positive] Sample 2", currentSample[1])

            # Into the batch:
            batchSamples.append([currentSample[0], currentSample[1]])
            # Pair label:
            batchLabels[i] = 1

        # Set the sample index:
        sampleIndex = len(batchSamples)

        # Get image size (using the first sample):
        imageHeight, imageWidth = batchSamples[0][0].shape[0:2]

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

                        # Show the random sample:
                        if displayImages:
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
            batchLabels[sampleIndex] = 0
            # Increase the "batch processing" index:
            sampleIndex += 1

        # Make sure to shuffle list of samples and labels:
        batchSamples, batchLabels = shuffleSamples(batchSamples, batchLabels)

        # python list To numpy array of numpy arrays...
        batchSamplesArray = np.array(batchSamples)

        image1Arrays = batchSamplesArray[:, 0:1]
        image2Arrays = batchSamplesArray[:, 1:2]

        # Reshape the goddamn arrays: (drop "list dimension"):
        tempDim = image1Arrays.shape

        image1Arrays = image1Arrays.reshape(tempDim[0], tempDim[2], tempDim[3], tempDim[4])
        image2Arrays = image2Arrays.reshape(tempDim[0], tempDim[2], tempDim[3], tempDim[4])

        # Show the batch:
        if displayImages:
            for h in range(6):
                print(h, batchLabels[h])
                showImage("[Batch] Sample 1", image1Arrays[h][0:imageHeight])
                showImage("[Batch] Sample 2", image2Arrays[h][0:imageHeight])

        outDict = {"image1": image1Arrays, "image2": image2Arrays}, batchLabels
        # outDict = {"image1": batch[:, 0], "image2": batch[:, 1:genresVectorLength + 1]}, batch[:, -1]
        yield outDict


# Set project paths:
projectPath = "D://dataSets//faces//"
outputPath = projectPath + "out//"
datasetPath = outputPath + "cropped"

# Script Options:
randomSeed = 42069

displayImages = False
displaySampleBatch = False

imageSize = (64, 64)
embeddingSize = 100

imagesPerClass = 60
pairsPerClass = 0.5 * (imagesPerClass ** 2.0) - (0.5 * imagesPerClass) - 7e-12
pairsPerClass = math.ceil(pairsPerClass)
# pairsPerClass = 1770  # 3321

modelFilename = "facenet.model"
loadModel = False
saveModel = False

# FaceNet training options:
lr = 0.001
trainingEpochs = 10
nPositive = 1024

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
# Store the samples total here:
totalDatasetSamples = 0

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

    # Load the full images (-1) or just the amount indicated (!= -1)
    if imagesPerClass != -1:
        imagesRange = imagesPerClass
    else:
        imagesRange = totalImages

    # Load the images:
    for p in range(imagesRange):
        # for currentPath in imagePaths:
        # Get current path:
        currentPath = imagePaths[p]
        # Load the image:
        currentImage = cv2.imread(currentPath)

        # Pre-process the image for FaceNet input:
        currentImage = cv2.resize(currentImage, imageSize)
        # Scale:
        currentImage = currentImage.astype("float") / 255.0

        # Show the input image:
        if displayImages:
            showImage("Input image [Pre-processed]", currentImage)

        # Into the list of samples:
        facesDataset[currentClass].append(currentImage)
        # Image counter goes up +1:
        totalDatasetSamples += 1

# Get total samples in dataset:
print("[FaceNet Training] Dataset Samples:", totalDatasetSamples)
print("Loaded: [" + str(imagesPerClass) + "] images per class.")

# Set random seed:
random.seed(randomSeed)
np.random.seed(randomSeed)

# Build the positive pairs dataset:
# Stores: (Class A - Sample 1, Class A - Sample 2, Class Code)
positivePairs = []

for currentClass in facesDataset:
    # Get images of current class:
    classImages = facesDataset[currentClass]
    # Shuffle images:
    random.shuffle(classImages)

    # Get total samples for this class:
    classSamples = len(classImages)

    # processed samples counter:
    processedPairs = 0
    for i in range(pairsPerClass):

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
            currentImage = classImages[sampleIndex]
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

# Shuffle the list of positive pairs:
random.shuffle(positivePairs)

# Check batch info:
if displaySampleBatch:

    # Generate sample batch:
    x, y = next(generateBatch(pairs=positivePairs, n_positive=5, negative_ratio=2))

    # Count total pos/neg samples:
    classCounters = [0, 0]

    # For every batch sample, get its pair and label and display the info in a
    # nice new window.
    for i, (label, img1, img2) in enumerate(zip(y, x["image1"], x["image2"])):

        # Set figure title and border:
        # Green border - positive pair
        # Red border - negative pair

        if label == 0:
            classText = "Negative"
            borderColor = (0, 0, 255)
            classCounters[1] += 1
        else:
            borderColor = (0, 255, 0)
            classText = "Positive"
            classCounters[0] += 1

        # Check the info:
        print(i, "Pair Label:", label, classText)

        # Get image dimensions:
        imageHeight, imageWidth = img1.shape[1:3]
        # Check the images:
        imageList = [img1[0:imageHeight], img2[0:imageHeight]]

        # Horizontally concatenate images:
        stackedImage = cv2.hconcat(imageList)
        (height, width) = stackedImage.shape[0:2]

        # Draw rectangle:
        cv2.rectangle(stackedImage, (0, 0), (width - 1, height - 1), borderColor, 1)

        # Show the positive/negative pair of images:
        showImage("[Generator] Sample", stackedImage)

    # Print the total count per class:
    print("Total Positives: ", classCounters[0], " Total Negatives: ", classCounters[1])

# Load or train model from scratch:
if loadModel:
    # Get model path + name:
    modelPath = projectPath + modelFilename
    print("[INFO] -- Loading faceNet Model from: " + modelPath)
    # Load model:
    model = load_model(modelPath)
    # Get summary:
    model.summary()
else:
    print("[INFO] -- Creating faceNet Model from scratch:")

    # Set the image dimensions:
    imageHeight = imageSize[0]
    imageWidth = imageSize[1]
    imageChannels = 3

    # Build the faceNet model:
    model = faceNet.build(height=imageHeight, width=imageWidth, depth=imageChannels, embeddingDim=embeddingSize,
                          alpha=lr, epochs=trainingEpochs)
    # Get faceNet summary:
    model.summary()

    # Plot faceNet model:
    graphPath = outputPath + "model_plot.png"
    plot_model(model, to_file=graphPath, show_shapes=True, show_layer_names=True)
    print("[INFO] -- Model graph saved to: " + graphPath)

    # # Set the samples' generator:
    gen = generateBatch(pairs=positivePairs, n_positive=nPositive, negative_ratio=2)

    # # Train the net:
    # # len(pairs) / 1024 = 754.68 (755)
    H = model.fit(gen,
                  # validation_split=0.2,
                  steps_per_epoch=len(positivePairs) // nPositive,
                  epochs=trainingEpochs,
                  verbose=1)

    print("Model Fitted...")
    #
    # # Check if model needs to be saved:
    # if saveModel:
    #     # Set model path:
    #     modelPath = outputPath + modelFilename
    #     print("[INFO] -- Saving model to: " + str(modelPath))
    #     model.save(modelPath)
    #
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
    plt.plot(N, H.history["accuracy"], label="train_acc")
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

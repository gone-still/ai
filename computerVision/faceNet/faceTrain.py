# File        :   faceTrain.py
# Version     :   0.8.9
# Description :   faceNet training script

# Date:       :   Jul 09, 2023
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT

import cv2
import math

from imutils import paths
from glob import glob
import numpy as np
import random

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

from faceNet import faceNet

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
def generateBatch(pairs, n_positive=2, negative_ratio=2, displayImages=False):
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

            # Scale data:
            # currentSample[0] = currentSample[0].astype("float") / 255.0
            # currentSample[1] = currentSample[1].astype("float") / 255.0

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

                        # Scale data:
                        # randomSample = randomSample.astype("float") / 255.0

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
datasetPath = outputPath + "cropped//train"

# Set the global random seed for all pseudo-random processes:
randomSeed = 420

# Debug:
displayImages = False
displaySampleBatch = False

# Script Options:
trainSplit = 0.8  # Dataset split for training
validationSize = -1  # Use this amount of samples from the validation split for validation, -1 uses the full validation split
validationStepsPercent = 1.0

# Generator generates this amount of positive pairs for training [0] and validation [1]:
nPositive = (128, 128)

# CNN image processing shape:
imageDims = (100, 100, 3)
resizeInterpolation = cv2.INTER_AREA

embeddingSize = 512
# Use this amount of images per class... -1 uses the whole available images per class,
# should be <= than the class with the least number of samples:
imagesPerClass = 55

# Vertically random-flip samples:
randomFlip = True
# Apply high-pass:
applyHighpass = True
# Randomly use grayscale:
randomGrayscale = False

# FaceNet training options:
# Choose sim metric: euclidean | cosine | sum
similarityMetric = "cosine"
lr = 0.001  # 0.0007
netParameters = {"euclidean": {"epochs": 30, "boundaries": [594, 2970], "values": [0.0035, 0.001, 0.0007]},
                 # "cosine": {"epochs": 30, "boundaries": [3468], "values": [0.0025, 0.001]},
                 "cosine": {"epochs": 15, "boundaries": [594, 1485], "values": [0.07, 0.0125, 0.0025]},
                 # "sum": {"epochs": 35, "boundaries": [2890], "values": [0.001, 0.001 * 0.6]}}
                 "sum": {"epochs": 20, "boundaries": [594, 1485], "values": [0.07, 0.0125, 0.0045]}}

# Create this amount of positive pairs...
# Extra pairs (not guaranteed to be unique):
extraPairs = 100
# Compute the max number of unique pairs:
pairsPerClass = (0.5 * (imagesPerClass ** 2.0) - (0.5 * imagesPerClass) - 7e-12) + extraPairs
pairsPerClass = math.ceil(pairsPerClass)

weightsFilename = "facenetWeights.h5"
loadWeights = False
saveWeights = True

# Print tf info:
print("[INFO - FaceNet Training] -- Tensorflow ver:", tf.__version__)

# Load each image path of the dataset:
print("[INFO - FaceNet Training] -- Loading images...")

# Set random seed:
random.seed(randomSeed)
np.random.seed(randomSeed)

# Store the samples total here:
totalDatasetSamples = 0

# The classes dictionary:
classesDictionary = {}

# Create the faces dataset as a dictionary:
facesDataset = {}

# Get list of full subdirectories paths:
classesDirectories = glob(datasetPath + "//*//", recursive=True)
classesDirectories.sort()

# Trim root directory, leave only subdirectories names (these will be the classes):
rootLength = len(datasetPath)
classesImages = [dirName[rootLength + 1:-1] for dirName in classesDirectories]

# Create classes dictionary:
classCounter = 0
for c in classesImages:
    if c not in classesDictionary:
        classesDictionary[c] = classCounter
        classCounter += 1

print("[INFO - FaceNet Training] -- Classes Dictionary:")
print(" ", classesDictionary)

# Get total classes:
totalClasses = len(classesImages)
print("[INFO - FaceNet Training] -- Total Classes:", totalClasses)

# Load images per class:
for c, currentDirectory in enumerate(classesDirectories):

    # Images for this class:
    imagePaths = list(paths.list_images(currentDirectory))
    totalImages = len(imagePaths)

    # Shuffle list:
    # random.shuffle(imagePaths)

    currentClass = classesImages[c]
    print(" ", "Class: " + currentClass + " Available Samples: " + str(totalImages))

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

        # Flip?
        flipInt = random.randint(0, 1)
        if flipInt == 1:
            # Flip along the y axis:
            currentImage = cv2.flip(currentImage, 1)

        # Use grayscale?
        if randomGrayscale:
            convertInt = random.randint(0, 4)
            if convertInt == 4:
                # To Gray
                currentImage = cv2.cvtColor(currentImage, cv2.COLOR_BGR2GRAY)
                # To BGR:
                currentImage = cv2.cvtColor(currentImage, cv2.COLOR_GRAY2BGR)

        # Should it be converted to grayscale (one channel):
        targetDepth = imageDims[-1]

        if targetDepth != 3:
            # To Gray:
            currentImage = cv2.cvtColor(currentImage, cv2.COLOR_BGR2GRAY)

        # Pre-process the image for FaceNet input:
        newSize = imageDims[0:2]

        # Resize:
        currentImage = cv2.resize(currentImage, newSize, resizeInterpolation)

        # Apply high-pass?
        if applyHighpass:
            kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
            currentImage = cv2.filter2D(currentImage, -1, kernel)

        # Scale:
        currentImage = currentImage.astype("float") / 255.0

        if targetDepth == 1:
            # Add "color" dimension:
            currentImage = np.expand_dims(currentImage, axis=-1)

        # Show the input image:
        if displayImages:
            showImage("Input image [Pre-processed]", currentImage)

        # Into the list of samples:
        facesDataset[currentClass].append(currentImage)
        # Image counter goes up +1:
        totalDatasetSamples += 1

    # Print the total of samples used:
    print(" ", "Class: " + currentClass + " Used Samples: " + str(imagesRange) + "/" + str(totalImages))

# Get total samples in dataset:
print("[INFO - FaceNet Training] -- Dataset Samples:", totalDatasetSamples)
print("[INFO - FaceNet Training] -- Loaded: [" + str(imagesPerClass) + "] images per class.")

# Build the positive pairs dataset:
# Stores: (Class A - Sample 1, Class A - Sample 2, Class Code)
print("[INFO - FaceNet Training] -- Generating: ", pairsPerClass, " pairs per class...")

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

        choices = list(range(0, classSamples))
        randomSamples = [0, 0]

        # Randomly choose a first sample:
        randomSamples[0] = random.choice(choices)

        # Randomly choose a second sample, excluding the one already
        # chosen:
        randomSamples[1] = random.choice(list(set(choices) - set([randomSamples[0]])))

        # Print the chosen samples:
        print(" ", "Processing pair: " + str(processedPairs), randomSamples)

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
    print("[" + currentClass + "]" + " Pairs Created: " + str(processedPairs) + ", Class: " + currentClass,
          "(" + str(classesDictionary[currentClass]) + ")", " Total: " + str(totalPairs))

# Shuffle the list of positive pairs:
random.shuffle(positivePairs)

# Split the pairs for train and validation,
# Training:
totalPairs = len(positivePairs)
trainSize = int(trainSplit * totalPairs)
print("[INFO - FaceNet Training] -- Train dataset has: " + str(trainSize) + " (" + str(
    trainSplit) + ") samples out of: " + str(totalPairs))

# Validation:
trainPairs = positivePairs[0:trainSize]
listSize = totalPairs - trainSize

if validationSize == -1:
    # Use all remaining samples:
    validationPairs = positivePairs[trainSize:]
else:
    # Slice the number of samples requested:
    listStart = trainSize
    listEnd = trainSize + validationSize
    validationPairs = positivePairs[listStart:listEnd]

# Check out some metrics:
validationSetSize = len(validationPairs)
validationSplit = validationSetSize / listSize

print("[INFO - FaceNet Training] -- Validation dataset has: " + str(validationSetSize) + " (" + str(
    validationSplit) + ") samples out of: " + str(
    listSize))

# Check batch info:
if displaySampleBatch:

    # Generate sample batch:
    batchesNames = ["Train", "Validation"]
    trainBatch = next(generateBatch(pairs=trainPairs, n_positive=5, negative_ratio=2, displayImages=False))
    validationBatch = next(generateBatch(pairs=validationPairs, n_positive=5, negative_ratio=2, displayImages=False))

    batchDataset = [trainBatch, validationBatch]

    for b in range(len(batchDataset)):
        currentBatch = batchesNames[b]
        x, y = batchDataset[b]

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
            print(" ", currentBatch, i, "Pair Label:", label, classText)

            # Get image dimensions:
            imageHeight, imageWidth = img1.shape[1:3]
            # Check the images:
            imageList = [img1[0:imageHeight], img2[0:imageHeight]]

            # Horizontally concatenate images:
            stackedImage = cv2.hconcat(imageList)
            imageDimensions = len(stackedImage.shape)

            if imageDimensions < 3:
                (height, width) = stackedImage.shape
                depth = 1
            else:
                (height, width, depth) = stackedImage.shape

            # Get image type:
            imageType = stackedImage.dtype

            # Check type and convert to uint8:
            if imageType != np.dtype("uint8"):
                stackedImage = stackedImage * 255.0
                stackedImage = stackedImage.clip(0, 255).astype(np.uint8)

                # Convert channels:
                if depth != 3:
                    stackedImage = cv2.cvtColor(stackedImage, cv2.COLOR_GRAY2BGR)

            # Draw rectangle:
            cv2.rectangle(stackedImage, (0, 0), (width - 1, height - 1), borderColor, 1)

            # Show the positive/negative pair of images:
            showImage("[" + currentBatch + " Generator] Sample", stackedImage)

        # Print the total count per class:
        print(" ", currentBatch, "Total Positives: ", classCounters[0], " Total Negatives: ", classCounters[1])

# Set the lr scheduler parameters:
print("[INFO - FaceNet Training] -- Distance set to: " + similarityMetric)
lrParameters = [netParameters[similarityMetric]["boundaries"], netParameters[similarityMetric]["values"]]

# Build the faceNet model:
model = faceNet.build(height=imageDims[0], width=imageDims[1], depth=imageDims[2], namesList=["image1", "image2"],
                      embeddingDim=embeddingSize, alpha=lr, distanceCode=similarityMetric,
                      lrSchedulerParameters=lrParameters)

# Load or train model from scratch:
if loadWeights:

    # Get model path + name:
    modelPath = outputPath + weightsFilename
    print("[INFO - FaceNet Training] -- Loading faceNet Model from: " + modelPath)
    # Load model:
    model.load_weights(modelPath)
    # Get summary:
    model.summary()

else:

    print("[INFO - FaceNet Training] -- Creating faceNet Model from scratch:")
    # Get faceNet summary:
    model.summary()

    # Plot faceNet model:
    graphPath = outputPath + "model_plot.png"
    plot_model(model, to_file=graphPath, show_shapes=True, show_layer_names=True)
    print("[INFO - FaceNet Training] -- Model graph saved to: " + graphPath)

    # Set the test/validation datasets portions:
    stepsPerEpoch = len(trainPairs) // nPositive[0]
    validationSteps = len(validationPairs) // nPositive[1]
    # validationSteps = int(validationStepsPercent * stepsPerEpoch)  # len(testPairs) // nPositive

    print("[INFO - FaceNet Training] -- Steps per epoch -> Training: " + str(stepsPerEpoch) + " Validation: " + str(
        validationSteps))

    # Set the samples' generator:
    trainGen = generateBatch(pairs=trainPairs, n_positive=nPositive[0], negative_ratio=2)
    validationGen = generateBatch(pairs=validationPairs, n_positive=nPositive[1], negative_ratio=2)

    # Train the net:
    trainingEpochs = netParameters[similarityMetric]["epochs"]
    H = model.fit(trainGen,
                  validation_data=validationGen,
                  steps_per_epoch=stepsPerEpoch,
                  validation_steps=validationSteps,
                  epochs=trainingEpochs,
                  verbose=1)

    # # Check if model needs to be saved:
    if saveWeights:
        # Set model path:
        modelPath = outputPath + weightsFilename
        print("[INFO - FaceNet Training] -- Saving model to: " + str(modelPath))
        # model.save(modelPath)
        model.save_weights(modelPath)

    # Plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()

    # Get the historical data:
    N = np.arange(0, trainingEpochs)

    history_dict = H.history
    print(history_dict.keys())

    # Plot values:
    plt.plot(N, H.history["loss"], label="train_loss")
    plt.plot(N, H.history["val_loss"], label="val_loss")
    plt.plot(N, H.history["accuracy"], label="train_acc")
    plt.plot(N, H.history["val_accuracy"], label="val_acc")
    # plt.plot(N, H.history["lr"], label="learning_rate")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")

    # Save plot to disk:
    plotPath = projectPath + "out//" + "lossGraph.png"
    print("[INFO - FaceNet Training] -- Saving model loss plot to:" + plotPath)
    plt.savefig(plotPath)
    plt.show()

# File        :   faceTrain.py
# Version     :   0.9.5
# Description :   faceNet training script

# Date:       :   Jul 26, 2023
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT

import cv2
import math

from imutils import paths
from glob import glob
import numpy as np
import random
import time
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau

from faceNet import faceNet
from faceConfig import getNetworkParameters

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
def shuffleSamples(batchSamples, batchLabels, randomSeed=42):
    # Get total size (rows) of the batch/matrix:
    batchSize = batchLabels.shape[0]
    # Create the ascending array of choices:
    choicesArray = np.arange(0, batchSize, 1, dtype=int)

    # Shuffle the choices array:
    np.random.seed(randomSeed)
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
def generateBatch(pairsList, totalSteps, batchSize=10, displayImages=False, randomIndex=False):
    # Get total number of pairs:
    totalPairs = len(pairsList)

    # Get total number of samples:
    totalSamples = batchSize * totalSteps

    # The numpy array of labels (positive=1, negative=-1)
    batchLabels = np.zeros((batchSize, 1))

    # Step counter:
    stepCounter = 0
    randomChoice = 0

    # Choices array:
    # if randomIndex:
    #     choicesArray = np.arange(0, totalPairs, 1, dtype=int)
    #     randomChoice = np.random.choice(choicesArray, 1, replace=False)[0]

    # This creates a generator, called by the neural network during
    # training...
    while True:
        # Check step counter:
        if stepCounter == totalSteps:
            stepCounter = 0
            # if randomIndex:
            #     randomChoice = np.random.choice(choicesArray, 1, replace=False)[0]

        # print(stepCounter)
        # print(" 1 Fuck ", stepCounter, randomChoice)

        # Store batch samples here:
        batchSamples = []

        # Store the positive random samples in the batch array:
        for i in range(batchSize):
            # Set list starting point:
            listIndex = (batchSize * stepCounter) + i
            if listIndex > (totalSamples - 1):
                listIndex = 0

            # Get current pair of images:
            listItem = pairsList[listIndex]
            currentSample = (listItem[0], listItem[1])

            # Check images (if proper data type):
            if displayImages:
                showImage("Batch Sample 1", currentSample[0])
                showImage("Batch Sample 2", currentSample[1])

            # Into the batch:
            batchSamples.append([currentSample[0], currentSample[1]])
            # Pair label:
            batchLabels[i] = listItem[2]

        # Make sure to shuffle list of samples and labels:
        batchSamples, batchLabels = shuffleSamples(batchSamples, batchLabels, int(time.time()))

        # python list To numpy array of numpy arrays...
        batchSamplesArray = np.array(batchSamples)

        image1Arrays = batchSamplesArray[:, 0:1]
        image2Arrays = batchSamplesArray[:, 1:2]

        # Reshape the goddamn arrays: (drop "list dimension"):
        tempDim = image1Arrays.shape

        image1Arrays = image1Arrays.reshape(tempDim[0], tempDim[2], tempDim[3], tempDim[4])
        image2Arrays = image2Arrays.reshape(tempDim[0], tempDim[2], tempDim[3], tempDim[4])

        # Get image size (using the first sample):
        imageHeight, imageWidth = pairsList[0][0].shape[0:2]

        # Increase step counter:
        stepCounter += 1

        # Show the batch:
        if displayImages:
            for h in range(tempDim[0]):
                print(h, batchLabels[h])
                showImage("[Batch] Sample 1", image1Arrays[h][0:imageHeight])
                showImage("[Batch] Sample 2", image2Arrays[h][0:imageHeight])

        outDict = {"image1": image1Arrays, "image2": image2Arrays}, batchLabels
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
loadWeights = False
saveWeights = True

# Generator generates this amount of positive pairs for training [0] and validation [1]:
# This is the "batch size" for positives:
nPositive = {"Train": 256, "Test": 256}

# Negative portion:
# Batch size: nPositive + negativeRatio * nPositive
negativeRatio = 1.0

# Get the network parameters:
configParameters = getNetworkParameters()

# CNN image processing shape:
imageDims = configParameters["imageDims"]

# Set interpolation type:
resizeInterpolation = cv2.INTER_AREA

embeddingSize = configParameters["embeddingSize"]
# Use this amount of images per class... -1 uses the whole available images per class,
# should be <= than the class with the least number of samples:
imagesPerClass = 55

# Vertically random-flip samples:
randomFlip = True
# Apply high-pass:
applyHighpass = configParameters["useHighPass"]
# Randomly use grayscale:
randomGrayscale = False

# FaceNet training options:
# Choose sim metric: euclidean | cosine | sum
similarityMetric = configParameters["similarityMetric"]

# Get the (fixed) learning rate:
lr = configParameters["lr"]

# Get the training configuration (epochs and lr scheduler config):
netParameters = configParameters["netParameters"]

# Create this amount of positive pairs...
# Extra pairs (not guaranteed to be unique):
extraPairs = 1485
# Compute the max number of unique pairs:
pairsPerClass = (0.5 * (imagesPerClass ** 2.0) - (0.5 * imagesPerClass) - 7e-12) + extraPairs
pairsPerClass = math.ceil(pairsPerClass)

# Set weights file name:
weightsFilename = "facenetWeights" + "-" + configParameters["weightsFilename"] + ".h5"

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
    random.shuffle(imagePaths)

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

        # Apply vertical Flip?
        if randomFlip:
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

    pairsCreated = {}

    for i in range(pairsPerClass):

        createPair = True

        while createPair:

            choices = list(range(0, classSamples))
            randomSamples = [0, 0]

            # Randomly choose a first sample:
            randomSamples[0] = random.choice(choices)

            # Randomly choose a second sample, excluding the one already
            # chosen:
            randomSamples[1] = random.choice(list(set(choices) - set([randomSamples[0]])))

            # Has the pair already been created?
            dictKey = "-".join(str(i) for i in randomSamples)
            if dictKey not in pairsCreated:
                # Register created pair:
                pairsCreated[dictKey] = True
                # Exit pair-creating loop:
                createPair = False
            # else:
            #     print(" ", "Pair: ", randomSamples,
            #           "(" + dictKey + ") has already been created, generating new pair...", len(pairsCreated))

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

    # Lemme sort the created pairs cause I wanna check it out...
    pairsCreated = dict(sorted(pairsCreated.items()))

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

# This Dict contains the full dataset: positive and negative samples
pairsDataset = {"Train": [], "Test": []}

# Set train/validation batch sizes:
# trainingDSSize = len(trainPairs) * (1 + negativeRatio)
# validationDSSize = len(validationPairs) * (1 + negativeRatio)
datasetData = {"Train": {"size": len(trainPairs), "list": trainPairs},
               "Test": {"size": len(validationPairs), "list": validationPairs}}

# Create the complete dataset -> train + validation:
for currentDataset in ["Train", "Test"]:

    # Set some local counters:
    samplesCounters = [0, 0]  # Positive, Negative

    # Get dataset size:
    currentSize = datasetData[currentDataset]["size"]
    # Get list:
    currentList = datasetData[currentDataset]["list"]

    print(currentDataset + " - Populating Dataset with [POSITIVE] pairs. [" + str(currentSize) + "]")

    # Process positive pairs:
    for i in range(currentSize):

        # Get current list item:
        listItem = currentList[i]  # Pair 1, Pair 2, Class
        # Get images:
        currentSamples = (listItem[0], listItem[1])
        # Check images (if proper data type):
        if displayImages:
            showImage("[Positive] Sample 1", currentSamples[0])
            showImage("[Positive] Sample 2", currentSamples[1])

        # Into the list -> Sample 1, Sample 2, Pair type (1-Pos|0-Neg)
        pairsDataset[currentDataset].append([currentSamples[0], currentSamples[1], 1])
        # Counter goes up:
        samplesCounters[0] += 1

    # Print the number of processed pairs so far:
    print(currentDataset + " - Total [POSITIVE] pairs stored:", samplesCounters[0])

    # Set array from where the random negative samples will be drawn from:
    choicesArray = np.arange(0, currentSize, 1, dtype=int)

    # Process negative pairs:
    totalNegatives = currentSize * negativeRatio
    # Round to the nearest even:
    totalNegatives = int(totalNegatives - totalNegatives % 2)

    print(currentDataset + " - Populating Dataset with [NEGATIVE] pairs. [" + str(totalNegatives) + "]")

    for j in range(totalNegatives):

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
                randomRow = currentList[randomChoice]

                # Get the sample's class:
                rowClass = randomRow[-1]
                # print(j, rowClass, pastClass)

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

        # This is a negative pair:
        tempList.append(0)

        # Got the two negative samples, thus the negative pair is ready:
        pairsDataset[currentDataset].append(tempList)
        # Counter goes up:
        samplesCounters[1] += 1

    # Print the number of processed pairs so far:
    print(currentDataset + " - Total [NEGATIVE] pairs stored:", samplesCounters[1])

    # Shake, shake, shake, Senora, shake your body line:
    # print([c[-1] for c in pairsDataset[currentDataset][0:5]])
    random.shuffle(pairsDataset[currentDataset])
    if currentDataset == "Test":
        for i in range(10):
            print("x ", i)
            random.shuffle(pairsDataset[currentDataset])
    # print([c[-1] for c in pairsDataset[currentDataset][0:5]])

print("Train pairs stored:", len(pairsDataset["Train"]))
print("Test pairs stored:", len(pairsDataset["Test"]))

# Check batch info:
if displaySampleBatch:

    # Generate sample batch:
    batchesNames = ["Train", "Validation"]
    randomDistribution = False

    trainBatch = next(generateBatch(pairsList=pairsDataset["Train"], totalSteps=1, batchSize=10, displayImages=False))
    validationBatch = next(
        generateBatch(pairsList=pairsDataset["Test"], totalSteps=1, batchSize=10, displayImages=False))

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
    stepsPerEpoch = len(pairsDataset["Train"]) // nPositive["Train"]
    validationSteps = len(pairsDataset["Test"]) // nPositive["Test"]
    # validationSteps = int(validationStepsPercent * stepsPerEpoch)  # len(testPairs) // nPositive

    print("[INFO - FaceNet Training] -- Steps per epoch -> Training: " + str(stepsPerEpoch) + " Validation: " + str(
        validationSteps))

    # Set the samples' generator:
    trainGen = generateBatch(pairsList=pairsDataset["Train"], totalSteps=stepsPerEpoch, batchSize=nPositive["Train"])
    validationGen = generateBatch(pairsList=pairsDataset["Test"], totalSteps=validationSteps,
                                  batchSize=nPositive["Test"])

    # Train the net:
    trainingEpochs = netParameters[similarityMetric]["epochs"]
    classWeights = configParameters["classWeights"]

    # LR reducer:
    reduceLr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3, mode="auto", min_delta=0.001, cooldown=0,
                                 min_lr=0.00001, verbose=1)
    # Model fit:
    H = model.fit(trainGen,
                  validation_data=validationGen,
                  class_weight=classWeights,
                  steps_per_epoch=stepsPerEpoch,
                  validation_steps=validationSteps,
                  epochs=trainingEpochs,
                  # callbacks=[reduceLr],
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

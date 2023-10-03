# File        :   faceTrain.py
# Version     :   0.10.X
# Description :   faceNet training script (Generator Version)

# Date:       :   Oct 2, 2023
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
from datetime import timedelta

import os

import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow import keras

from faceNet import faceNet
from DataGenerator import DataGenerator
from faceConfig import getNetworkParameters

from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.optimizers import RMSprop

import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import Callback


# Reads an image:
def readImage(imagePath):
    # Loads image:
    inputImage = cv2.imread(imagePath)
    # Checks if image was successfully loaded:
    if inputImage is None:
        print("Image Path: ", imagePath)
        raise TypeError("readImage>> Error: Could not load Input image.")

    return inputImage


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


# Rotates an image by a given angle (degs):
def rotateImage(inputImage, angle):
    # Grab the dimensions of the image and calculate the center of the
    # image
    (h, w) = inputImage.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # rotate our image by 45 degrees around the center of the image
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    rotatedImage = cv2.warpAffine(inputImage, M, (w, h))
    return rotatedImage


# Shuffles a batch of lists (images) and a numpy array (labels) that
# share a "row" in a matrix:
def shuffleSamples(batchSamples, batchLabels, randomSeed=42):
    # Get total size (rows) of the batch/matrix:
    batchSize = len(batchLabels)
    # Create the ascending array of choices:
    choicesArray = np.arange(0, batchSize, 1, dtype=int)

    # Shuffle the choices array:
    np.random.seed(randomSeed)
    np.random.shuffle(choicesArray)

    # Output containers:
    shuffledList = []
    shuffledArray = []

    # Shuffle the actual list and array:
    for i in range(len(choicesArray)):
        # Grab random choice:
        randomNumber = choicesArray[i]
        # Grab image pair from list and
        # label from array, imagePair
        # is also a list:
        shuffledList.append(batchSamples[randomNumber])
        shuffledArray.append(batchLabels[randomNumber])

    return shuffledList, shuffledArray


# Batch generator of positive & negatives pairs:
def generateBatch(pairsList, totalSteps, batchSize=10, displayImages=False, randomIndex=False):
    # Get total number of pairs:
    totalPairs = len(pairsList)

    # Get total number of samples:
    totalSamples = batchSize * totalSteps

    # The numpy array of labels (positive=1, negative=-1)
    batchLabels = np.zeros((batchSize, 1), dtype="float32")

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
        del batchSamples[:]
        del batchSamples

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


# Callback that displays learning rate at each epoch end:


class LearningRateTracker(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Get current optimizer used by the model:
        optimizer = self.model.optimizer
        # Get learning rate:

        if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
            _lr = optimizer._optimizer._decayed_lr("float32").numpy()
        else:
            _lr = optimizer._decayed_lr("float32").numpy()

        # _lr = optimizer._decayed_lr(tf.float32).numpy()
        print(f"Epoch {epoch + 1}: Learning rate is: ", _lr)
        lines = ["Epoch: " + str(epoch + 1) + " Lr: " + str(_lr) + " " + str(logs)]
        txtPath = outputPath + "logger.txt"
        with open(txtPath, "a") as f:
            for line in lines:
                f.write(line)
                f.write('\n')


# Image Pre-processing function:
def imagePreprocessing(inputImage, newSize, optionsDict, auxfunDict, verbose=False):
    # Unpack the dict of options:
    applyRotation = optionsDict["Rotation"]
    randomFlip = optionsDict["Flip"]
    randomGrayscale = optionsDict["Grayscale"]
    applyHighpass = optionsDict["HighPass"]

    # Aux Functions:
    rotationFunction = auxfunDict["RotateFun"]

    # Apply small rotation?
    if applyRotation:
        rotationAngle = random.randint(-10, 10)
        inputImage = rotationFunction(inputImage, rotationAngle)
        # print("imagePreprocessing>> Applied rotation of: "+str(rotationAngle))

    # Apply vertical Flip?
    if randomFlip:
        flipInt = random.randint(0, 1)
        if flipInt == 1:
            # Flip along the y axis:
            inputImage = cv2.flip(inputImage, 1)

    # Use grayscale?
    if randomGrayscale:
        convertInt = random.randint(0, 4)
        if convertInt == 4:
            # To Gray
            inputImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
            # To BGR:
            inputImage = cv2.cvtColor(inputImage, cv2.COLOR_GRAY2BGR)

    # Should it be converted to grayscale (one channel):
    targetDepth = imageDims[-1]

    if targetDepth != 3:
        # To Gray:
        inputImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

    # Pre-process the image for FaceNet input:
    # Resize:
    inputImage = cv2.resize(inputImage, newSize, resizeInterpolation)

    # Apply high-pass?
    if applyHighpass:
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        inputImage = cv2.filter2D(inputImage, -1, kernel)

    # Scale:
    inputImage = inputImage.astype("float32") / 255.0

    if targetDepth == 1:
        # Add "color" dimension:
        inputImage = np.expand_dims(inputImage, axis=-1)

    # Show the input image:
    if verbose:
        showImage("Input image [Pre-processed]", inputImage)

    return inputImage


# Set project paths:
projectPath = "D://dataSets//faces//"
outputPath = projectPath + "out//"
datasetPath = outputPath + "cropped//train"

# Set the global random seed for all pseudo-random processes:
randomSeed = 42069

# Class labels for positive and negative pairs:
classLabels = {"Positive": 1, "Negative": 0}

# Include unique pairs?
includeUniques = True
# Display unique pairs?
displayUniques = False
# How many unique pairs must be included,
# -1 Uses all the unique images available:
totalUniques = -1

# Skip images from this class:
excludedClasses = ["Uniques"]

# Debug:
displayImages = False
displaySampleBatch = False

# Script Options:
trainSplit = 0.8  # Dataset split for training
validationSize = -1  # Use this amount of samples from the validation split for validation, -1 uses the full validation split
validationStepsPercent = 1.0
loadWeights = False
resumeTraining = False

saveWeights = True
outmodelName = "faceNet-siam"

# Default starting epoch (0):
startingEpoch = 0

# Negative portion:
# Batch size: nPositive + negativeRatio * nPositive
negativeRatio = 1.0

# Get the network parameters:
configParameters = getNetworkParameters()

# Generator generates this amount of positive pairs for training [0] and validation [1]:
# This is the "batch size" for positives:
nPositive = {"Train": configParameters["batchSize"]["Train"],
             "Test": configParameters["batchSize"]["Test"]}

print("Batch Sizes: Train - " + str(configParameters["batchSize"]["Train"]) + " Test - " + str(
    configParameters["batchSize"]["Test"]))

# CNN image processing shape:
imageDims = configParameters["imageDims"]

# Set interpolation type:
resizeInterpolation = cv2.INTER_AREA

embeddingSize = configParameters["embeddingSize"]
# Use this amount of images per class... -1 uses the whole available images per class,
# should be <= than the class with the least number of samples:
imagesPerClass = 2

# Set pre-processing options:
optionsDict = {"Rotation": False, "Flip": True, "Grayscale": False, "HighPass": configParameters["useHighPass"]}

# Auxiliary pre-processing functions
auxfunDict = {"RotateFun": rotateImage}

# FaceNet training options:
# Choose sim metric: euclidean | cosine | sum
similarityMetric = configParameters["similarityMetric"]

# Get the (fixed) learning rate:
lr = configParameters["lr"]

# Get the regularization factor:
regFactor = configParameters["regFactor"]

# Get the dropout factor:
dropFactor = configParameters["dropFactor"]

# Get the training configuration (epochs and lr scheduler config):
netParameters = configParameters["netParameters"]

# Create this amount of positive pairs...
# Extra pairs (not guaranteed to be unique):
extraPairs = 0
# Compute the max number of unique pairs:
pairsPerClass = (0.5 * (imagesPerClass ** 2.0) - (0.5 * imagesPerClass) - 7e-12) + extraPairs
pairsPerClass = math.ceil(pairsPerClass)

# Set weights file name:
weightsFilename = "facenetWeights" + "-" + configParameters["weightsFilename"] + ".h5"

# Set checkpoint path:
checkpointPath = outputPath + "checkpoint//"

# Print tf info:
print("[INFO - FaceNet Training] -- Tensorflow ver:", tf.__version__)

# Set mixed precision:
# keras.mixed_precision.set_global_policy("mixed_float16")

# config = tf.ConfigProto(device_count={"CPU": 8})
# keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

# Explicitly set GPU as computing device:
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.set_visible_devices(physical_devices[0], "GPU")

# Load each image path of the dataset:
print("[INFO - FaceNet Training] -- Loading images...")

# Set random seed:
# tf.random.set_seed(randomSeed)
random.seed(randomSeed)
np.random.seed(randomSeed)

# The classes dictionary:
classesDictionary = {}

# Get list of full subdirectories paths:
classesDirectories = [f.path for f in os.scandir(datasetPath) if f.is_dir()]
# classesDirectories = glob(datasetPath + "//*//", recursive=True)
classesDirectories.sort()

# Trim root directory, leave only subdirectories names (these will be the classes):
rootLength = len(datasetPath)

# Filter the classes:
if excludedClasses:
    filteredList = []
    for currentDir in classesDirectories:
        # Get dir/class name:
        className = currentDir[rootLength + 1:-1]
        # Check if dir/class must be filtered:
        if className not in excludedClasses:
            # Into the filtered list:
            filteredList.append(currentDir)
        else:
            print("[INFO - FaceNet Training] -- Filtered class/dir: " + className)

    # Filtered list is now
    classesDirectories = filteredList

# Strip name classes from directory paths:
classesImages = [os.path.split(dirName)[-1] for dirName in classesDirectories]

# Create classes dictionary (Key = className, Value = classCode):
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

# Split images into two sets:
trainClasses = math.ceil(trainSplit * totalClasses)
testClasses = totalClasses - trainClasses
imageDataset = {"Train": {"Paths": [], "Split": trainClasses, "Start": 0},
                "Test": {"Paths": [], "Split": testClasses, "Start": trainClasses}}

# Set array from where the random negative samples will be drawn from:
choicesArray = np.arange(0, totalClasses, 1, dtype=int)
# Samplin without replacement, as the path can only be selected once:
choicesArray = np.random.choice(choicesArray, len(choicesArray), replace=False)

# Path counter for possible duplicated paths:
counterDict = {}

for currentDataset in imageDataset:
    # Get dataset split:
    currentSplit = imageDataset[currentDataset]["Split"]
    # Get starting iterator offset:
    startOffset = imageDataset[currentDataset]["Start"]
    i = 0

    # Get paths for this dataset:
    while i < currentSplit:
        # Get current choice:
        arrayIndex = i + startOffset
        randomChoice = choicesArray[arrayIndex]
        randomPath = classesDirectories[randomChoice]

        # Into the image dataset:
        imageDataset[currentDataset]["Paths"].append(randomPath)
        i += 1

        # Beware that the path is unique, not duplicated:
        if randomPath not in counterDict:
            counterDict[randomPath] = True
        else:
            # For some reason, there were duplicated entries:
            raise TypeError("Found duplicated path")

# Create the faces dataset as a dictionary:
facesDataset = {"Train": {}, "Test": {}}

# Store the samples total here:

totalDatasetSamples = {"Train": 0, "Test": 0}

# Load images per class:
for currentDataset in facesDataset:
    # Get possible path list for this dataset:
    pathList = imageDataset[currentDataset]["Paths"]
    totalPaths = len(pathList)
    print("Dataset: " + currentDataset + ", Available Classes: " + str(totalPaths))

    for c, currentDirectory in enumerate(pathList):

        # Images for this class:
        imagePaths = list(paths.list_images(currentDirectory))
        totalImages = len(imagePaths)

        # Shuffle list:
        random.shuffle(imagePaths)

        # Get current class, which is the last part of the split
        # currentDirectory string:
        currentClass = os.path.split(currentDirectory)[-1]
        classCode = classesDictionary[currentClass]

        print(" ", "Class: " + currentClass + " (" + str(classCode) + ") " + "Available Samples: " + str(
            totalImages) + " [" + str(c + 1) + "/" + str(totalPaths) + "]")

        # Create dictionary key:
        # Each key is a class name associated with
        # an array of samples for this class:
        facesDataset[currentDataset][currentClass] = []

        # Load the full images (-1) or just the amount indicated (!= -1)
        if imagesPerClass != -1:
            imagesRange = imagesPerClass
        else:
            imagesRange = totalImages

        # Load the paths to images::
        for p in range(imagesRange):
            # Get current path:
            currentPath = imagePaths[p]

            # Into the list of sample paths:
            facesDataset[currentDataset][currentClass].append(currentPath)

            # Path counter goes up +1:
            totalDatasetSamples[currentDataset] += 1

        # Print the total of samples used:
        print(" ", "Class: " + currentClass + " Used Samples: " + str(imagesRange) + "/" + str(totalImages))

# Get total samples in dataset:
print("[INFO - FaceNet Training] -- Dataset Samples:", totalDatasetSamples["Train"] + totalDatasetSamples["Test"])
print("[INFO - FaceNet Training] -- Train Samples:", totalDatasetSamples["Train"])
print("[INFO - FaceNet Training] -- Test Samples:", totalDatasetSamples["Test"])
print("[INFO - FaceNet Training] -- Loaded: " + str(imagesPerClass) + " images per class.")

# Build the positive pairs dataset:
# Stores: (Class A - Sample 1, Class A - Sample 2, Class Code)
print("[INFO - FaceNet Training] -- Generating: ", pairsPerClass, " pairs per class...")

# Positive pairs are stored here:
positivePairs = {"Train": [], "Test": []}

for currentDataset in facesDataset:
    # Get current source list (images list):
    classList = facesDataset[currentDataset]

    for currentClass in classList:

        # Get paths of current class:
        classPaths = classList[currentClass]

        # Shuffle sample paths:
        for i in range(10):
            random.shuffle(classPaths)

        # Get total samples for this class:
        classSamples = len(classPaths)

        # processed samples counter:
        processedPairs = 0

        pairsCreated = {}
        classCode = -1

        # Store the current pair here:
        currentPair = [0, 0]

        # Local pair counter:
        pairCount = 0

        for i in range(classSamples - 1):

            # Get first class sample:
            firstPath = classPaths[i]

            # Log pair index:
            currentPair[0] = i

            # Show first image
            if displayImages:
                sampleImage = readImage(firstPath)
                showImage(currentClass + "Current Pair {A}", sampleImage)

            for j in range(i + 1, classSamples):

                # Store sample pair here:
                tempList = []
                # First sample goes into the temp list:
                tempList.append(firstPath)

                # Get second class sample:
                secondPath = classPaths[j]
                # Into the temp list:
                tempList.append(secondPath)

                # Finally, store class code:
                classCode = classesDictionary[currentClass]
                tempList.append(classCode)

                # Into the positive pairs list:
                positivePairs[currentDataset].append(tempList)
                processedPairs += 1

                # Log pair index:
                currentPair[1] = j

                # Log created positive pair:
                dictKey = "-".join(str(i) for i in currentPair)

                if dictKey not in pairsCreated:
                    # Register created pair:
                    pairsCreated[dictKey] = True
                    # Exit pair-creating loop:
                    createPair = False
                    print("Class: ", currentClass, "Pair: ", currentPair)
                else:
                    # For some reason, the same pair was built more than once:
                    raise TypeError("Class:, ", currentClass, " Pair: ", currentPair,
                                    "(" + dictKey + ") has already been created")

                # Show second image
                if displayImages:
                    currentImage = readImage(secondPath)
                    showImage(currentClass + "Current Pair {B}", currentImage)

                # Show the whole pair:
                if displayImages:
                    # Load images from paths:
                    firstImage = readImage(tempList[0])
                    secondImage = readImage(tempList[1])

                    # Preprocess images:
                    firstImage = imagePreprocessing(firstImage, imageDims[0:2], optionsDict, auxfunDict, False)
                    secondImage = imagePreprocessing(secondImage, imageDims[0:2], optionsDict, auxfunDict, False)

                    # Horizontally concatenate images:
                    imageList = [firstImage, secondImage]
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

                    # Show image:
                    showImage("Unique Pair", stackedImage)

                # Pair counter goes up:
                pairCount += 1

        # Lemme sort the created pairs cause I wanna check it out...
        pairsCreated = dict(sorted(pairsCreated.items()))

        # Done creating positive pairs for this class:
        totalPairs = len(positivePairs)
        print("[" + currentClass + "]" + " Pairs Created: " + str(
            processedPairs) + ", Class: " + currentClass + "(" + str(
            classesDictionary[currentClass]) + ")",
              " Total: " + str(totalPairs) + " [Sub-total: " + str(pairCount) + "]")

# Process "Uniques":
uniquePairs = {"Train": [], "Test": []}

if includeUniques:

    # Max class code:
    classCode = len(classesDictionary)

    # Create new class coded in the classes dictionary:
    classesDictionary["Uniques"] = classCode

    print("Processing Unique pairs...")

    # Set the Uniques directory:
    uniquesDirectory = os.path.join(datasetPath, "Uniques")

    # Images for this class:
    imagePaths = list(paths.list_images(uniquesDirectory))

    # Total unique pairs:
    totalImages = len(imagePaths)

    # Set total of images to be used:
    if totalUniques == -1:
        totalUniques = totalImages

    # Store sample pair here:
    tempList = []

    # Get the filename name mnemonic:
    tempString = imagePaths[0].split(".")
    tempString = tempString[0].split("//")
    tempString = tempString[-1]
    filename = tempString.split("-")

    # All uniques are loaded to this temp list:
    tempUniquesList = []

    uniqueCount = 0

    # Create the unique pairs:
    createUniques = True
    i = 0
    # Log loading time:
    start = time.time()
    while createUniques:

        # for i, currentPath in enumerate(imagePaths):

        # Set the image name:
        if i % 2 == 0:
            lastChar = "A"
        else:
            lastChar = "B"

        # Create complete path:
        currentPath = imagePaths[i]
        print("Current Unique image: ", currentPath)

        # Load the image:
        # currentImage = cv2.imread(currentPath)
        # Pre-process the image:
        # currentImage = imagePreprocessing(currentImage, imageDims[0:2], optionsDict, auxfunDict, displayImages)

        # Into the temp list:
        if lastChar == "A":
            tempList.append(currentPath)
        else:
            tempList.append(currentPath)
            # Finally, store class code:
            tempList.append(classCode)
            classCode += 1

            # Into the positive pairs list:
            tempUniquesList.append(tempList)
            uniqueCount += 1

            if displayUniques:
                # Load images:
                firstImage = readImage(tempList[0])
                secondImage = readImage(tempList[1])

                # Preprocess images:
                firstImage = imagePreprocessing(firstImage, imageDims[0:2], optionsDict, auxfunDict, False)
                secondImage = imagePreprocessing(secondImage, imageDims[0:2], optionsDict, auxfunDict, False)

                # Horizontally concatenate images:
                imageList = [firstImage, secondImage]
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

                # Show image:
                showImage("Unique Pair", stackedImage)

            # Clear list:
            tempList = []

        # Check loop control variables:
        i = i + 1
        if i >= totalUniques:
            createUniques = False

    # Show loading time:
    elapsed = (time.time() - start)
    elapsed = str(timedelta(seconds=elapsed))
    print("-> Unique loading time: ", elapsed)

    # Shuffle the list of unique pairs:
    random.shuffle(tempUniquesList)

    # Done creating positive pairs for this class:
    totalPairs = len(tempUniquesList)
    print("[Unique]" + " Pairs Created: " + str(uniqueCount), "(" + str(uniqueCount * 2) + " Images)",
          "(Class Code: " + str(classesDictionary["Uniques"]) + ")", " Total: " + str(totalPairs))

    # Get the split sizes:
    totalUniquePairs = int(0.5 * totalUniques)
    trainUniques = math.ceil(trainSplit * totalUniquePairs)
    testUniques = totalUniquePairs - trainUniques

    print("Train Uniques: " + str(trainUniques) + " pairs.")
    print("Test Uniques: " + str(testUniques) + " pairs.")

    # Split train and test uniques:
    uniquePairs["Train"] = tempUniquesList[0:trainUniques]
    uniquePairs["Test"] = tempUniquesList[trainUniques:]

    print("Train/Test pair count: ", len(uniquePairs["Train"]), len(uniquePairs["Test"]))
    del tempUniquesList

# Shuffle the list of positive pairs:
for currentDataset in positivePairs:
    for i in range(10):
        random.shuffle(positivePairs[currentDataset])

# Prepare final train and test lists of positive pairs:
trainPairs = positivePairs["Train"] + uniquePairs["Train"]
validationPairs = positivePairs["Test"] + uniquePairs["Test"]

trainSize = len(trainPairs)
validationSize = len(validationPairs)
totalPairs = trainSize + validationSize

print("[INFO - FaceNet Training] -- Train dataset has: " + str(trainSize) + " (" + str(
    trainSplit) + ") samples out of: " + str(totalPairs))
print("[INFO - FaceNet Training] -- Validation dataset has: " + str(validationSize) + " (" + str(
    (1.0 - trainSplit)) + ") samples out of: " + str(totalPairs))

# This Dict contains the full dataset: positive and negative samples
pairsDataset = {"Train": {"Samples": [], "Labels": []},
                "Test": {"Samples": [], "Labels": []}}

# Set train/validation batch sizes:
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
        # Get class:
        sampleClass = listItem[2]
        # Check images (if proper data type):
        if displayImages:
            # Load images:
            firstSample = readImage(currentSamples[0])
            secondSample = readImage(currentSamples[1])
            # Show images:
            showImage("[Positive] Sample 1", firstSample)
            showImage("[Positive] Sample 2", secondSample)

        # Into the list -> Sample 1, Sample 2, Pair type (Pos|Neg)
        # pairsDataset[currentDataset].append([currentSamples[0], currentSamples[1], classLabels["Positive"]])
        pairsDataset[currentDataset]["Samples"].append([currentSamples[0], currentSamples[1]])
        pairsDataset[currentDataset]["Labels"].append(classLabels["Positive"])

        # Counter goes up:
        samplesCounters[0] += 1

    # Print the number of processed pairs so far:
    print(currentDataset + " - Total [POSITIVE] pairs stored:", samplesCounters[0])

    # Set array from where the random negative samples will be drawn from:
    choicesArray = np.arange(0, currentSize, 1, dtype=int)
    choicesArray = np.random.choice(choicesArray, len(choicesArray), replace=False)

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
            randomIndex = j
            while getRows:
                # Get a random row from the positive pairs list,
                # "Flatten" to an integer, since the return value is
                # an array:
                # randomChoice = np.random.choice(choicesArray, 1)[0]
                # print(randomIndex)
                randomChoice = choicesArray[randomIndex]

                # Actually Get the random row:
                randomRow = currentList[randomChoice]

                # Get the sample's class:
                rowClass = randomRow[-1]
                # print("Class: ", rowClass)
                # print(j, rowClass, pastClass)

                if rowClass != pastClass:
                    # Randomly choose one of the two images:
                    randomChoice = random.randint(0, 1)
                    randomSample = randomRow[randomChoice]

                    # Scale data:
                    # randomSample = randomSample.astype("float") / 255.0

                    # Show the random sample:
                    if displayImages:
                        # Load image:
                        currentSample = readImage(randomSample)
                        # Show image:
                        showImage("randomSample: " + str(s), currentSample)

                    # Into the temp list:
                    tempList.append(randomSample)
                    # Stop the loop:
                    getRows = False

                    # Present is now past:
                    pastClass = rowClass

                else:
                    # Select another row, via the random index:
                    if randomIndex < len(choicesArray) - 1:
                        randomIndex += 1
                    else:
                        randomIndex = 0

        # This is a negative pair:
        tempList.append(classLabels["Negative"])

        # Got the two negative samples, thus the negative pair is ready:
        pairsDataset[currentDataset]["Samples"].append(tempList[0:2])
        pairsDataset[currentDataset]["Labels"].append(tempList[2])

        # Counter goes up:
        samplesCounters[1] += 1

    # Print the number of processed pairs so far:
    print(currentDataset + " - Total [NEGATIVE] pairs stored:", samplesCounters[1])

    # Shake, shake, shake, Senora, shake your body line:
    # print([c[-1] for c in pairsDataset[currentDataset][0:5]])
    # random.shuffle(pairsDataset[currentDataset])
    # if currentDataset == "Test":
    for i in range(10):
        pairsDataset[currentDataset]["Samples"], pairsDataset[currentDataset]["Labels"] = shuffleSamples(
            pairsDataset[currentDataset]["Samples"], pairsDataset[currentDataset]["Labels"], randomSeed=randomSeed)
    # print([c[-1] for c in pairsDataset[currentDataset][0:5]])

print("Train pairs stored:", len(pairsDataset["Train"]["Samples"]))
print("Test pairs stored:", len(pairsDataset["Test"]["Samples"]))

# Set the preprocessing options:
# Config -> Configure the operations in preprocessing pipeline
# AuxFuns -> Dict of auxiliary functions used by the preprocessing pipeline
preprocessingConfig = {"config": optionsDict, "auxFuns": auxfunDict}

# Check batch info:
if displaySampleBatch:

    # Generate sample batch:
    batchesNames = ["Train", "Validation"]
    randomDistribution = False

    trainBatch = DataGenerator(pairsDataset["Train"]["Samples"], pairsDataset["Train"]["Labels"], imagePreprocessing,
                               preprocessingConfig, debug=False, batchSize=32, imgSize=imageDims[0:2], shuffle=False)
    validationBatch = DataGenerator(pairsDataset["Test"]["Samples"], pairsDataset["Test"]["Labels"], imagePreprocessing,
                                    preprocessingConfig, debug=False, batchSize=32, imgSize=imageDims[0:2],
                                    shuffle=False)

    # Nastily call a pair of test batches:
    trainBatch = trainBatch.__getitem__(0)
    validationBatch = validationBatch.__getitem__(0)

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

            if label == classLabels["Negative"]:
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
optimizer = Adamax
model = faceNet.build(height=imageDims[0], width=imageDims[1], depth=imageDims[2], namesList=["image1", "image2"],
                      optimizer=optimizer,
                      embeddingDim=embeddingSize, alpha=lr, distanceCode=similarityMetric,
                      lrSchedulerParameters=[], regFactor=regFactor, dropFactor=dropFactor, lossMargin=1)

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

    if resumeTraining:

        from tensorflow.keras.utils import CustomObjectScope
        from tensorflow.keras.models import load_model

        print("[INFO - FaceNet Training] -- Resuming training...")

        # Check model file:
        modelPath = checkpointPath + "bestSoFar.keras"
        fileExists = os.path.exists(modelPath)
        if not fileExists:
            raise ValueError("Model file: " + modelPath + " does not exist.")

        # # Check checkpoint file:
        # fileExists = os.path.exists(checkpointPath)
        # if not fileExists:
        #     raise ("Checkpoint file: " + checkpointPath + " does not exist.")

        from ContrastiveLoss import ContrastiveLoss
        from WeightedAverage import WeightedAverage
        from EuclideanDistance import EuclideanDistance

        model = load_model(modelPath, custom_objects={"ContrastiveLoss": ContrastiveLoss,
                                                      "WeightedAverage": WeightedAverage,
                                                      "EuclideanDistance": EuclideanDistance})

        # Get resuming epoch:
        startingEpoch = configParameters["startingEpoch"]

        print(
            "[INFO - FaceNet Training] -- Loaded Model from: " + modelPath + ". Starting Epoch: " + str(startingEpoch))

    else:

        print("[INFO - FaceNet Training] -- Creating faceNet Model from scratch:")

    # Get faceNet summary:
    model.summary()

    # Plot faceNet model:
    graphPath = outputPath + "model_plot.png"
    plot_model(model, to_file=graphPath, show_shapes=True, show_layer_names=True)
    print("[INFO - FaceNet Training] -- Model graph saved to: " + graphPath)

    # Set the test/validation datasets portions:
    stepsPerEpoch = len(pairsDataset["Train"]["Samples"]) // nPositive["Train"]
    validationSteps = len(pairsDataset["Test"]["Samples"]) // nPositive["Test"]
    # validationSteps = int(validationStepsPercent * stepsPerEpoch)  # len(testPairs) // nPositive

    print("[INFO - FaceNet Training] -- Steps per epoch -> Training: " + str(stepsPerEpoch) + " Validation: " + str(
        validationSteps))

    # Set the samples' generator:
    trainGen = DataGenerator(pairsDataset["Train"]["Samples"], pairsDataset["Train"]["Labels"], imagePreprocessing,
                             preprocessingConfig, debug=False, batchSize=nPositive["Train"], imgSize=imageDims[0:2],
                             shuffle=False, name="train")

    validationGen = DataGenerator(pairsDataset["Test"]["Samples"], pairsDataset["Test"]["Labels"], imagePreprocessing,
                                  preprocessingConfig, debug=False, batchSize=nPositive["Test"], imgSize=imageDims[0:2],
                                  shuffle=False, name="validations")

    # Train the net:
    trainingEpochs = netParameters[similarityMetric]["epochs"]
    classWeights = configParameters["classWeights"]

    # LR reducer:
    reduceLr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, mode="auto", verbose=1, cooldown=1,
                                 min_lr=0.00125, min_delta=0.0015)

    # Model Checkpoint,
    # Model weights are saved at the end of every epoch, if it's the best seen so far:
    checkpointFilename = "bestSoFar.keras"  # "{epoch:02d}-{val_accuracy:.2f}-{val_loss:.2f}.h5"
    modelCheckpoint = ModelCheckpoint(filepath=checkpointPath + checkpointFilename,
                                      save_weights_only=False,
                                      monitor="val_accuracy",
                                      mode="max", save_best_only=True,
                                      verbose=0)

    # Model fit:
    H = model.fit(x=trainGen,
                  validation_data=validationGen,
                  initial_epoch=startingEpoch,
                  class_weight=classWeights,
                  # steps_per_epoch=stepsPerEpoch,
                  # validation_steps=validationSteps,
                  epochs=trainingEpochs,
                  # callbacks=[LearningRateTracker(), reduceLr, modelCheckpoint],
                  callbacks=[LearningRateTracker(), reduceLr],
                  workers=10,
                  max_queue_size=10,
                  use_multiprocessing=False,
                  verbose=1,
                  shuffle=True)

    # H = model.fit_generator(generator=trainGen,
    #                         validation_data=validationGen,
    #                         callbacks=[LearningRateTracker(), reduceLr],
    #                         workers=8,
    #                         max_queue_size=20,
    #                         use_multiprocessing=False,
    #                         verbose=1,
    #                         shuffle=True)

    # Check if model needs to be saved:
    if saveWeights:
        # Set model path:
        modelPath = outputPath + weightsFilename
        print("[INFO - FaceNet Training] -- Saving model to: " + str(modelPath))

        # Save full model (architecture + weights + optimizer state):
        saveFormats = ["tf", "keras"]
        for currentFormat in saveFormats:
            outName = outputPath + outmodelName + "." + currentFormat
            print("[INFO - FaceNet Training] -- Saving: " + outName)
            model.save(outName, save_format=saveFormats, overwrite=True)

    # Plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()

    # Get the historical data:
    historyEpochs = trainingEpochs - startingEpoch
    N = np.arange(0, historyEpochs)

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

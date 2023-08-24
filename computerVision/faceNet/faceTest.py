# File        :   faceTest.py
# Version     :   0.11.6
# Description :   faceNet test script

# Date:       :   Aug 11,2023
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT

import cv2
import math
import os

import tensorflow as tf

from imutils import paths
from glob import glob
import numpy as np

import random
import time

from faceConfig import getNetworkParameters
from faceNet import faceNet

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score
from matplotlib import pyplot as plt


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


# Natural sorts a list of strings:
def naturalSort(inputList, prefix, imgExt=".png"):  # image_

    # Attempting to naturally sort the list of strings:
    tempList = []
    filenameCounter = 0
    prefixLen = len(prefix)
    for name in inputList:
        # Get the number from string:
        tempString = name[prefixLen:]
        tempString = tempString.split(".")
        fileNumber = tempString[0].split("//")
        fileNumber = fileNumber[-1]

        # Store in temp list:
        tempList.append(fileNumber)

    # Sort the list (ascending)
    tempList.sort()
    # Prepare the (ordered) output list of strings:
    outList = [prefix + str(n) + imgExt for n in tempList]

    return outList


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


# Set project paths:
projectPath = "D://dataSets//faces//"
outputPath = projectPath + "out//"
datasetPath = outputPath + "cropped//test"

# Get the network parameters:
configParameters = getNetworkParameters()

# Set distance metric:
similarityMetric = configParameters["similarityMetric"]

# Set weights file name:
weightsFilename = "facenetWeights" + "-" + configParameters["weightsFilename"] + ".h5"

# Write folder:
resultsPath = outputPath + "results//cosine18//"

# Total positive pairs & negative pairs to be tested:
# startImage = 55  # Start at this image for test
maxImages = 30  # Use all remaining images for test (-1 uses all the images)
datasetSize = 350
positivePortion = 0.7

randomSeed = 42069
pairsPerClass = 200

displayImages = False
displayUniques = False
showClassified = False
includeUniques = True
writeAll = False
randomFlip = True
applyRotation = False

# Print tf info:
print("[INFO - FaceNet Testing] -- Tensorflow ver:", tf.__version__)

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[0], 'GPU')

# Skip images from this class:
excludedClasses = ["Uniques"]

# Apply high-pass:
applyHighpass = configParameters["useHighPass"]

# Set interpolation type:
resizeInterpolation = cv2.INTER_AREA

# Set the DNN's parameters:
imageDims = configParameters["imageDims"]
embeddingSize = configParameters["embeddingSize"]
lr = configParameters["lr"]

# Get the training configuration (epochs and lr scheduler config):
netParameters = configParameters["netParameters"]
lrParameters = [netParameters[similarityMetric]["boundaries"], netParameters[similarityMetric]["values"]]

# Get the training configuration (epochs and lr scheduler config):
netParameters = configParameters["netParameters"]

# Set the image dimensions:
imageHeight = imageDims[0]
imageWidth = imageDims[1]
imageChannels = imageDims[2]

# Set random seed:
random.seed(randomSeed)
np.random.seed(randomSeed)

# Read the test images:
# Get list of full subdirectories paths:
classesDirectories = glob(datasetPath + "//*//", recursive=True)
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
            print("[INFO - FaceNet Testing] -- Filtered class/dir: " + className)

    # Filtered list is now
    classesDirectories = filteredList

# Load the classes:
classesImages = [dirName[rootLength + 1:-1] for dirName in classesDirectories]

# The classes dictionary:
classesDictionary = {}

# Create classes dictionary:
classCounter = 0
for c in classesImages:
    if c not in classesDictionary:
        classesDictionary[c] = classCounter
        classCounter += 1

print("[INFO - FaceNet Testing] -- Classes Dictionary:")
print(" ", classesDictionary)

# Get total classes:
totalClasses = len(classesImages)
print("[INFO - FaceNet Testing] -- Total Classes:", totalClasses)

# Store the samples total here:
totalDatasetSamples = 0
imagesRange = 0

# Create the test dataset as a dictionary:
testDataset = {}

# Load images per class:
for c, currentDirectory in enumerate(classesDirectories):
    # Get class:
    currentClass = classesImages[c]

    # Images for this class:
    imagePaths = list(paths.list_images(currentDirectory))
    # Shake the hat:
    random.shuffle(imagePaths)
    totalImages = len(imagePaths)

    # Slice test images:
    if maxImages != -1:
        # Check for minimum number of necessary images to create
        # a unique pair:
        if totalImages < 2:
            print(
                "[INFO - FaceNet Testing] -- Skipping class: " + currentClass + " due to insufficient samples to build pairs. " "Test Images: " + str(
                    totalImages))
            continue

        # More images than needed, trim the range:
        if totalImages >= maxImages:
            # Set the image range:
            endImage = maxImages
        # Not enough images:
        else:
            endImage = totalImages

        # Get the requested paths:
        imagePaths = imagePaths[0:endImage]

    testSamples = len(imagePaths)

    print("[INFO - FaceNet Testing] -- Class: " + currentClass + ", using: " + str(testSamples) + " samples for "
                                                                                                  "testing...")

    # Create dictionary key:
    # Each key is a class name associated with
    # an array of samples for this class:
    testDataset[currentClass] = []

    # Load the full images:
    imagesRange = testSamples

    # Load the images:
    for p in range(imagesRange):
        # for currentPath in imagePaths:
        # Get current path:
        currentPath = imagePaths[p]

        # Load the image:
        currentImage = cv2.imread(currentPath)

        # Apply small rotation:
        if applyRotation:
            rotationAngle = random.randint(-10, 10)
            currentImage = rotateImage(currentImage, rotationAngle)

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

        # Apply vertical Flip?
        if randomFlip:
            flipInt = random.randint(0, 1)
            if flipInt == 1:
                # Flip along the y axis:
                currentImage = cv2.flip(currentImage, 1)

        # Scale:
        currentImage = currentImage.astype("float") / 255.0

        if targetDepth == 1:
            # Add "color" dimension:
            currentImage = np.expand_dims(currentImage, axis=-1)

        # Show the input image:
        if displayImages:
            showImage("Input image [Pre-processed]", currentImage)

        # Into the list of samples:
        testDataset[currentClass].append(currentImage)
        # Image counter goes up +1:
        totalDatasetSamples += 1

# Get total samples in dataset:
print("[INFO - FaceNet Testing] -- Dataset Samples:", totalDatasetSamples)
print(" ", "Loaded: [" + str(imagesRange) + "] images per class.")

# Create the positive pairs:
if pairsPerClass == -1:
    pairsPerClass = 0.5 * (imagesRange ** 2.0) - (0.5 * imagesRange) - 7e-12
    pairsPerClass = math.ceil(pairsPerClass)

print("[INFO - FaceNet Testing] -- Creating: " + str(pairsPerClass) + " pairs per class...")

# Positive pairs are stored here:
positivePairs = []

for currentClass in testDataset:
    # Get images of current class:
    classImages = testDataset[currentClass]

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

# Process "Uniques":
uniquePairs = []
if includeUniques:

    # Max class code:
    classCode = len(classesDictionary)
    # Create new class coded in the classes dictionary:
    classesDictionary["Uniques"] = classCode

    print("Processing Unique pairs...")

    # Set the Uniques directory:
    uniquesDirectory = datasetPath + "//Uniques//"
    # Images for this class:
    imagePaths = list(paths.list_images(uniquesDirectory))
    # Total unique pairs:
    totalImages = len(imagePaths)

    # Store sample pair here:
    tempList = []

    # Get the filename name mnemonic:
    tempString = imagePaths[0].split(".")
    tempString = tempString[0].split("//")
    tempString = tempString[-1]
    filename = tempString.split("-")

    uniqueCount = 0

    # Check the images:
    for i, currentPath in enumerate(imagePaths):
        # Set the image name:
        if i % 2 == 0:
            lastChar = "A"
        else:
            lastChar = "B"

        # Create complete path:
        # imageName = filename[0] + "-" + str(i + 1) + "-" + lastChar
        # currentPath = uniquesDirectory + imageName + ".png"
        print("Current Unique image: ", currentPath)

        # Load the image:
        currentImage = cv2.imread(currentPath)

        # Apply small rotation:
        if applyRotation:
            rotationAngle = random.randint(-10, 10)
            currentImage = rotateImage(currentImage, rotationAngle)

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

        # Apply vertical Flip?
        if randomFlip:
            flipInt = random.randint(0, 1)
            if flipInt == 1:
                # Flip along the y axis:
                currentImage = cv2.flip(currentImage, 1)

        # Scale:
        currentImage = currentImage.astype("float") / 255.0

        if targetDepth == 1:
            # Add "color" dimension:
            currentImage = np.expand_dims(currentImage, axis=-1)

        # Show the input image:
        if displayImages:
            showImage("[Unique Pair] Input image [Pre-processed]", currentImage)

        # Into the temp list:
        if lastChar == "A":
            tempList.append(currentImage)
        else:
            tempList.append(currentImage)
            # Finally, store class code:
            tempList.append(classCode)
            classCode += 1

            # Into the positive pairs list:
            uniquePairs.append(tempList)
            uniqueCount += 1

            # Clear list:
            tempList = []

            if displayUniques:

                # Horizontally concatenate images:
                imageList = [uniquePairs[-1][0], uniquePairs[-1][1]]
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

    # Shuffle the list of unique pairs:
    random.shuffle(uniquePairs)

    # Done creating positive pairs for this class:
    totalPairs = len(uniquePairs)
    print("[Unique]" + " Pairs Created: " + str(uniqueCount), "(" + str(uniqueCount * 2) + " Images)",
          "(Class Code: " + str(classesDictionary["Uniques"]) + ")", " Total: " + str(totalPairs))

# Shuffle the list of positive pairs:
random.shuffle(positivePairs)

# Create the "test batch":
testBatch = []  # List of pairs and real labels

# Compute the dataset portions:
totalUniques = 0
if not uniquePairs:
    # No unique pairs are present, just directly get
    # dataset portions:
    totalPositives = int(positivePortion * datasetSize)
    totalNegatives = datasetSize - totalPositives

else:
    # Get total of unique pairs:
    totalUniques = len(uniquePairs)

    # print("[INFO - FaceNet Testing] -- Storing unique pairs for test batch...")
    # for i in range(totalUniques):
    #
    #     # Get current pair of images:
    #     currentSample = uniquePairs[i]
    #
    #     # Check images (if proper data type):
    #     if displayImages:
    #         showImage("[Unique] Sample 1", currentSample[0])
    #         showImage("[Unique] Sample 2", currentSample[1])
    #
    #     # Into the batch - Pair and label (1):
    #     testBatch.append(([currentSample[0], currentSample[1]], 1))

    # Get positive and negative dataset portions:
    totalNegatives = int((1 - positivePortion) * datasetSize)
    totalPositives = datasetSize - (totalNegatives + totalUniques)

print("[INFO - FaceNet Testing] -- Positive Samples: " + str(totalPositives) + " Negative Samples: " + str(
    totalNegatives) + " Unique Samples: " + str(totalUniques) + " Total: " + str(
    totalPositives + totalNegatives + totalUniques))

# Get total number of pairs:
totalPairs = len(positivePairs)

# Randomly choose n positive examples from the pairs list:
choicesArray = np.arange(0, totalPairs, 1, dtype=int)
positiveSamples = np.random.choice(choicesArray, totalPositives, replace=True)

# Store the positive random pairs in the batch array:
print("[INFO - FaceNet Testing] -- Storing positive pairs for test batch...")
for i in range(totalPositives):

    # Get current pair of images:
    randomSample = positiveSamples[i]
    currentSample = positivePairs[randomSample]

    # Check images (if proper data type):
    if displayImages:
        showImage("[Positive] Sample 1", currentSample[0])
        showImage("[Positive] Sample 2", currentSample[1])

    # Into the batch - Pair and label (1):
    testBatch.append(([currentSample[0], currentSample[1]], 1))

positiveBatchSize = len(testBatch)
print("[INFO - FaceNet Testing] -- Positive pairs stored: " + str(positiveBatchSize))

# Now, include unique pairs:
if includeUniques:

    for i in range(totalUniques):

        # Get current pair of images:
        currentSample = uniquePairs[i]

        # Check images (if proper data type):
        if displayImages:
            showImage("[Positive] Sample 1", currentSample[0])
            showImage("[Positive] Sample 2", currentSample[1])

        # Into the batch - Pair and label (1):
        testBatch.append(([currentSample[0], currentSample[1]], 1))

    print("[INFO - FaceNet Testing] -- Unique pairs stored: " + str(totalUniques))

    # Include unique pairs in positive pairs list:
    positivePairs = positivePairs + uniquePairs
    random.shuffle(positivePairs)
    random.shuffle(positivePairs)

    # Re-roll the options:
    totalPairs = len(positivePairs)
    choicesArray = np.arange(0, totalPairs, 1, dtype=int)

# Store the negative random pairs:
print("[INFO - FaceNet Testing] -- Storing negative pairs for test batch...")

for i in range(totalNegatives):

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
            randomRow = positivePairs[randomChoice]

            # Get the sample's class:
            rowClass = randomRow[-1]
            # print(randomChoice, rowClass)

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
    testBatch.append((tempList, 0))

negativeBatchSize = len(testBatch) - positiveBatchSize
print("[INFO - FaceNet Testing] -- Negative pairs stored: " + str(negativeBatchSize))

# Shuffle list:
random.shuffle(testBatch)
print("[INFO - FaceNet Testing] -- Total pair samples in batch: " + str(len(testBatch)))

# Build the faceNet model:

model = faceNet.build(height=imageHeight, width=imageWidth, depth=imageChannels, namesList=["image1", "image2"],
                      embeddingDim=embeddingSize, alpha=lr, distanceCode=similarityMetric,
                      lrSchedulerParameters=lrParameters)

# Load in weights:
weightsFilePath = outputPath + weightsFilename
print("[INFO - FaceNet Testing] -- Loading faceNet weights file from: " + weightsFilePath)
model.load_weights(weightsFilePath)

# Get summary:
model.summary()

# Check if output directory must be created:
if writeAll:
    print("[INFO - FaceNet Testing] -- Checking output directory: " + resultsPath)
    directoryExists = os.path.isdir(resultsPath)
    if not directoryExists:
        print("[INFO - FaceNet Testing] -- Creating Directory: " + resultsPath)
        os.mkdir(resultsPath)
        directoryExists = os.path.isdir(resultsPath)
        if directoryExists:
            print("[INFO - FaceNet Testing] -- Successfully created directory: " + resultsPath)
    else:
        print("[INFO - FaceNet Testing] -- Directory Found.")

# Prepare predictions and real labels lists:
yPred = []
yTest = []

# Image writer counter:
imageCounter = 0

# Test the batch:
for b in range(len(testBatch)):
    # Get current pair & label:
    testPair = testBatch[b][0]
    testLabel = testBatch[b][1]

    # Add "batch" dimension:
    image1 = np.expand_dims(testPair[0], axis=0)
    image2 = np.expand_dims(testPair[1], axis=0)

    # Pack into dict:
    tempDict = {"image1": image1, "image2": image2}

    # Send to CNN, get probability:
    predictions = model.predict(tempDict)

    # Get current prediction:
    currentPrediction = predictions[0][0]

    # Set label color:
    if currentPrediction < 0.5:
        predictedLabel = 0
        classText = "Negative"
        borderColor = (0, 0, 255)
    else:
        predictedLabel = 1
        borderColor = (0, 255, 0)
        classText = "Positive"

    # Check the info:
    text = str(testLabel) + " : " + f'{currentPrediction:.4f}'
    print(" ", b, "Class: ", classText, " Predicted: ", f'{currentPrediction:.4f}', " Real: ", testLabel)

    # Store the info for CF plotting:
    yTest.append(testLabel)
    # Store the info for CF plotting:
    yPred.append(predictedLabel)

    # Compose pair image
    # Check the images:
    imageList = [testPair[0], testPair[1]]

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

    # Resize image:
    imgHeight, imgWidth = stackedImage.shape[0:2]
    aspectRatio = imgHeight / imgWidth

    # Up-scaling using original aspect ratio:
    displaySize = 150
    newWidth = int(displaySize // aspectRatio)

    stackedImage = cv2.resize(stackedImage, (newWidth, displaySize), interpolation=cv2.INTER_LINEAR)

    # Create text strip:
    stripHeight = int(0.2 * displaySize)
    stripShape = (stripHeight, newWidth, 3)
    textStrip = np.zeros(shape=stripShape, dtype="uint8")

    # Text & color:
    if predictedLabel == testLabel:
        if testLabel == 1:
            textColor = (0, 255, 0)
        else:
            textColor = (0, 0, 255)
    else:
        text = text + " [x]"
        textColor = (0, 128, 255)

    cv2.putText(textStrip, text, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=textColor, thickness=1)
    # showImage("text strip", textStrip)

    # Vertically concatenate images:
    stackedImage = cv2.vconcat([stackedImage, textStrip])

    # Show the positive/negative pair of images:
    key = -1
    if showClassified:
        windowName = "Test Sample"
        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
        cv2.imshow(windowName, stackedImage)
        key = cv2.waitKey(0)

    if writeAll or key == ord("e"):  # Press "e" to save image:
        print("[INFO - FaceNet Testing] -- Saving image to disk...")
        imagePath = resultsPath + "siameseResult_" + str(imageCounter) + ".png"
        writeImage(imagePath, stackedImage)
        imageCounter += 1

# Compute precision & recall:
modelPrecision = precision_score(yTest, yPred)
modelRecall = recall_score(yTest, yPred)

# Print the results:
dateNow = time.strftime("%Y-%m-%d %H:%M")
print(" ---------------------------------------------------------- ")
print("[INFO - FaceNet Testing] -- Test time: " + dateNow)
print("[INFO - FaceNet Testing] -- Used model: {" + weightsFilename + "}")
print("[INFO - FaceNet Testing] -- Wrote to: " + resultsPath)
print("[INFO - FaceNet Testing] -- Params - includeUniques:", includeUniques, "positivePortion:", positivePortion)
print("[INFO - FaceNet Testing] -- Computing Precision and Recall:")
print((modelPrecision, modelRecall))

# Compute and print confusion matrix:
print("[INFO - FaceNet Testing] -- Computing Confusion Matrix:")
result = confusion_matrix(yTest, yPred)  # normalize='pred'
print(result)
accuracy = (result[0][0] + result[1][1]) / datasetSize
print("Accuracy:", accuracy)

disp = ConfusionMatrixDisplay(confusion_matrix=result)
disp.plot()
plt.show()

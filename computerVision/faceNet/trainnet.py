import cv2
import os
from imutils import paths
from glob import glob

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


# Set project paths:
projectPath = "D://dataSets//faces//"
datasetPath = projectPath + "out//cropped"

# Script Options:
displayImages = False
imageSize = (50, 50)
pairsPerClass = 15

# Load each image path of the dataset:
print("[FaceNet Training] Loading images...")

# Get list of full subdirectories paths:
classesDirectories = glob(datasetPath + "//*//", recursive=True)
classesDirectories.sort()

# Trim root directory, leave only subdirectories names (these will be the classes):
rootLength = len(datasetPath)
classesImages = [dirName[rootLength + 1:-1] for dirName in classesDirectories]

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
random.seed(420)

# Build the positive pairs dataset:
positivePairs = []

for currentClass in facesDataset:
    # Get total samples for this class:
    classSamples = len(facesDataset[currentClass])

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
            currentImage = facesDataset[currentClass][sampleIndex]
            # Show image
            if True:
                showImage(currentClass + " Pair: " + str(s), currentImage)
            # Into the temp list:
            tempList.append(currentImage)

        # Into the positive pairs list:
        positivePairs.append(tempList)
        processedPairs += 1

    # Done creating positive pairs for this class:
    totalPairs = len(positivePairs)
    print("Pairs Created: " + str(processedPairs) + ", Class: " + currentClass, ", Total: " + str(totalPairs))

# File        :   samplesOrganizer.py
# Version     :   0.0.1
# Description :   Train/Test sample organizer.
#                 Moves sample files between source/target directories.
# Date:       :   Jul 28, 2023
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT

import random

import cv2
import os
import math

from imutils import paths
from glob import glob


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
projectPath = "D://dataSets//faces//out//cropped//"

# Train & Test directories:
trainPath = projectPath + "Train" + "//"
testPath = projectPath + "Test" + "//"

# Set directory paths and amount of sample files,
# -1 Sets remaining samples:
# testSamples = totalSamples - trainSamples
dirParams = {0: {"Name": "Train", "Path": trainPath, "Samples": 0, "Requested": 25},
             1: {"Name": "Test", "Path": testPath, "Samples": 0, "Requested": -1}}

# Read classes (directory names) from
# Train & Test directories:
dirClasses = {"Train": {}, "Test": {}}

for currentDir in dirParams:

    # Get dataset/directory name:
    currentDataset = dirParams[currentDir]
    datasetName = currentDataset["Name"]
    # Get its path:
    currentPath = currentDataset["Path"]

    # Get list of full subdirectories paths:
    classesDirectories = glob(currentPath + "//*//", recursive=True)
    classesDirectories.sort()

    print("\n""[INFO] Processing path: ", currentPath, " Classes: ", len(classesDirectories), "\n")

    # Trim root directory, leave only subdirectories names (these will be the classes):
    rootLength = len(currentPath)

    # Get the classes names (directories) for this dataset:
    classesNames = [dirName[rootLength - 1:-1] for dirName in classesDirectories]

    # Store class name & path in dictionary:
    for i, currentClass in enumerate(classesNames):
        # Look for entry in dictionary
        if currentClass not in dirClasses[datasetName]:
            # Class is key, path is value:
            dirClasses[datasetName][currentClass] = classesDirectories[i]
            print("", i, currentClass, dirClasses[datasetName][currentClass])

    # Store total samples for this dataset:
    currentDataset["Samples"] = len(dirClasses[datasetName])
    # Print some info:
    print("[INFO] " + datasetName + " total samples: " + str(currentDataset["Samples"]))

tempDict = {"Source": {}, "Target": {}}
maxSamples = 0

# Set source & target directories,
# Check amount of samples to be transferred/received
# Source -> Sends samples
# Target <- Receives samples
# Source has more than dirParams[datasetName]["Samples"]
# Target has less than dirParams[datasetName]["Samples"]
setDirectories = False
for currentDir in dirParams:
    # Get dataset/directory name:
    currentDataset = dirParams[currentDir]
    # Get name:
    datasetName = currentDataset["Name"]
    # Get sample count after/before transferring:
    sampleCount = currentDataset["Requested"]
    # Get total samples for this dataset:
    totalSamples = currentDataset["Samples"]

    if sampleCount != -1:
        print("Dataset: " + str(datasetName) + " is going to be source/target directory.")

        # Dataset has more samples than needed:
        if totalSamples > sampleCount:
            if currentDir == 0:
                sourceDir = 0
                targetDir = 1
            else:
                sourceDir = 1
                targetDir = 0
        # Dataset needs more samples than currently has:
        else:
            if currentDir == 0:
                sourceDir = 1
                targetDir = 0
            else:
                sourceDir = 0
                targetDir = 1

        # Set the source (sender):
        tempDict["Source"] = dirParams[sourceDir]
        sourceName = dirParams[sourceDir]["Name"]
        sourceSamples = dirParams[sourceDir]["Samples"]
        sourceCount = dirParams[sourceDir]["Requested"]
        # Check if the sender can send the amount of samples:
        samplesDiff = sourceSamples - sampleCount
        if samplesDiff < 0:
            raise TypeError("Source directory [" + sourceName + "] has: ", sourceSamples, " but Target requested: ",
                            sampleCount)
        sampleTotal = sourceSamples - sampleCount
        print("Dataset:", sourceName, "is Source:", sourceSamples, "(-) ->", sourceCount,
              "[Final: " + str(sampleTotal) + "]")

        # Set the target (receiver):
        tempDict["Target"] = dirParams[targetDir]
        targetName = dirParams[targetDir]["Name"]
        targetSamples = dirParams[targetDir]["Samples"]
        targetCount = dirParams[targetDir]["Requested"]
        sampleTotal = targetSamples + sampleCount
        print("Dataset:", targetName, "is Target:", targetSamples, "<- (+)", targetCount,
              "[Final: " + str(sampleTotal) + "]")

        # Dirs have been set:
        setDirectories = True

        break

# Check that directories have been set before attempting file transfer:
if not setDirectories:
    raise TypeError("Source/Target directories have not been set!")

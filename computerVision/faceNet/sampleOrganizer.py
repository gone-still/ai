# File        :   samplesOrganizer.py
# Version     :   0.3.0
# Description :   Train/Test sample organizer.
#                 Moves sample files between source/target directories.
# Date:       :   Aug 01, 2023
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT

import random

import cv2
import os
import math

from imutils import paths
from glob import glob

from natsort import os_sorted
import shutil


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


def getSourceTarget(currentSamples, samplesFinal, currentDir, dirParams, dirCodesReverse):
    sampleDiff = currentSamples - samplesFinal
    # Dataset has more samples than needed:
    if currentSamples > samplesFinal:
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

    sourceName = dirParams[sourceDir]["Name"]
    targetName = dirParams[targetDir]["Name"]

    print(sourceName + " -> Source")
    print(targetName + " -> Target")

    reverseDict = {targetDir: "Target", sourceDir: "Source"}

    outDict = {"Source": [dirParams[sourceDir]["Path"], 0],
               "Target": [dirParams[targetDir]["Path"], 0]}

    # Check that source has enough samples to move to target:
    samplesMoved = currentSamples - samplesFinal
    if samplesMoved < 0:
        raise TypeError("Source directory [" + datasetName + "] has: ", currentSamples,
                        " but Target requested: ", samplesFinal)

    if sampleDiff > 0:
        sign = 1
    else:
        sign = -1

    # This amount of samples are gonna be transferred over to target:
    outDict[reverseDict[targetDir]][1] = sign * sampleDiff
    # This amount of samples are retrieved from source:
    outDict[reverseDict[sourceDir]][1] = -sign * sampleDiff

    print(outDict)

    return outDict


# Set project paths:
projectPath = "D://dataSets//faces//out//cropped//"

# Train & Test directories:
trainPath = projectPath + "Train" + "//"
testPath = projectPath + "Test" + "//"

# Skip images from this class:
excludedClasses = ["Uniques"]

finalSamples = {"Train": 25, "Test": -1}

# Set directory paths and amount of sample files,
# -1 Sets remaining samples:
# testSamples = totalSamples - trainSamples
dirParams = {0: {"Name": "Train", "Path": trainPath, "availableSamples": 0, "samplesMoved": 0,
                 "finalSamples": finalSamples["Train"]},
             1: {"Name": "Test", "Path": testPath, "availableSamples": 0, "samplesMoved": 0,
                 "finalSamples": finalSamples["Test"]}}

# Read classes (directory names) from
# Train & Test directories:
dirClasses = {"Train": {}, "Test": {}}
dirCodes = {"Train": 0, "Test": 1}
dirCodesReverse = {0: "Train", 1: "Test"}

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

    # Filter the classes:
    if excludedClasses:
        filteredList = []
        for currentDir in classesDirectories:
            # Get dir/class name:
            className = currentDir[rootLength - 1:-1]
            # Check if dir/class must be filtered:
            if className not in excludedClasses:
                # Into the filtered list:
                filteredList.append(currentDir)
            else:
                print("Filtered class/dir: " + className)

        # Filtered list is now
        classesDirectories = filteredList

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
    currentDataset["availableSamples"] = len(dirClasses[datasetName])
    # Print some info:
    print("[INFO] " + datasetName + " total samples: " + str(currentDataset["availableSamples"]))

dirDictionary = {"Source": "", "Target": ""}
maxSamples = 0

# Set source & target directories,
# Check amount of samples to be transferred/received
# Source -> Sends samples
# Target <- Receives samples
# Source has more than dirParams[datasetName]["availableSamples"]
# Target has less than dirParams[datasetName]["availableSamples"]
setDirectories = False

for currentDir in dirParams:
    # Get dataset/directory name:
    currentDataset = dirParams[currentDir]
    # Get name:
    datasetName = currentDataset["Name"]
    # Get sample count after/before transferring:
    samplesFinal = currentDataset["finalSamples"]
    # Get total samples for this dataset:
    currentSamples = currentDataset["availableSamples"]
    # Get dataset path:
    currentPath = currentDataset["Path"]

    if samplesFinal != -1:
        # Dirs have been set:
        setDirectories = True


# Check that directories have been set before attempting file transfer:
if not setDirectories:
    raise TypeError("Source/Target directories have not been set!")

# Actually move the samples between directories:
# Get classes por each dir:

sourceClasses = dirClasses["Train"]
targetClasses = dirClasses["Test"]

for currentClass in sourceClasses:
    # Check if this class exists in target directory:
    if currentClass in targetClasses:

        # Get path for this class:
        sourcePath = sourceClasses[currentClass]
        # targetPath = targetClasses[currentClass]
        # print(sourcePath, targetPath)

        # Images for this class:
        # sourceImagePaths = list(paths.list_images(sourcePath))
        # Natural-sort the list, os style:
        # sourceImagePaths = os_sorted(sourceImagePaths)
        totalSourceImages = len(list(paths.list_images(sourcePath)))

        # Get codes:
        sourceCode = dirCodes["Train"]
        targetCode = dirCodes["Test"]

        # Get the info:
        samplesFinal = finalSamples["Train"]
        infoDict = getSourceTarget(totalSourceImages, samplesFinal, sourceCode, dirParams, dirCodesReverse)

        # Target and Source paths:
        sourcePath = infoDict["Source"][0] + currentClass
        targetPath = infoDict["Target"][0] + currentClass

        # Amount of samples to transfer from source to target
        samplesToMove = abs(infoDict["Source"][1])

        # Images for this class:
        sourceImagePaths = list(paths.list_images(sourcePath))
        # Natural-sort the list, os style:
        sourceImagePaths = os_sorted(sourceImagePaths)
        totalSourceImages = len(sourceImagePaths)

        # Image Counter:
        movedSamples = 0

        # Move the samples
        print("Class: " + currentClass + " found in Source and Target directories. Moving: " + str(samplesToMove) +
              " samples...")

        # Move the samples from source -> target:
        for s in range(samplesToMove):
            # Get reverse sample index:
            sampleIndex = (totalSourceImages - s) - 1
            # Look for sample path on images list:
            sourceSamplePath = sourceImagePaths[sampleIndex]

            # Get filename:
            fileName = sourceSamplePath.split("\\")[-1]

            # Set target path:
            targetSamplePath = targetPath + "//" + fileName

            # # check if file exist in destination
            if os.path.exists(targetSamplePath):
                print("File: " + fileName + " exists on: " + targetSamplePath + ". Renaming new file...")
                # Split name and extension
                fileName = fileName.split(".")
                # Adding the new name
                newName = fileName[0] + '_new' + fileName[1]
                # New Path:
                targetSamplePath = targetSamplePath + newName

            # Move file:
            print("Moving: " + sourceSamplePath + " --> " + targetSamplePath)
            shutil.move(sourceSamplePath, targetSamplePath)
            movedSamples += 1

        print("Class: " + currentClass + ", Samples Moved: " + str(movedSamples))

        # Get new directory total files:
        totalSourceImages = len(list(paths.list_images(sourcePath)))
        totalTargetImages = len(list(paths.list_images(targetPath)))

        print("Source images: " + str(totalSourceImages) + " Target images: " + str(totalTargetImages))

    else:
        print("Class: " + currentClass + " not found in Target directory, skipping...")

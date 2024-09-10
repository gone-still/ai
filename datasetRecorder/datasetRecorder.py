# File        :   datasetRecorder.py
# Version     :   0.5.1
# Description :   Records the state of a dataset (samples in validation split +training split)

# Date:       :   Sept 10, 2024
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT

import cv2
import os

import random
import math

import ast

from pathlib import Path


def isEven(number: int) -> bool:
    """
    Checks if a number is even
    :param number: the number to check
    :return: bool, True if even, False if odd
    """
    if number % 2 == 0:
        return True
    else:
        return False


def readImage(imagePath):
    """
    Reads image via OpenCV
    """
    # Loads image:
    inputImage = cv2.imread(imagePath)
    # Checks if image was successfully loaded:
    if inputImage is None:
        raise ValueError("readImage>> Error: Could not load Input image.")
    return inputImage


def showImage(imageName, inputImage):
    """
    Shows an image in a OpenCV High GUI window
    """
    cv2.namedWindow(imageName, cv2.WINDOW_NORMAL)
    cv2.imshow(imageName, inputImage)
    cv2.waitKey(0)


def writeImage(imagePath, inputImage):
    """
    Writes an png image
    :param imagePath: image path as a string
    :param inputImage: image as a numpy array
    :return: Void
    """
    cv2.imwrite(imagePath, inputImage, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print("writeImage>> Wrote Image: " + imagePath)


def writeList2File(filePath: str, outputList: list) -> None:
    """
    Writes a list of strings to a text file:
    :param filePath: path of the text file
    :param outputList: List of strings to write
    :return: Void
    """
    # Open file in write mode:
    file = open(filePath, "w")

    # Process tuples:
    for currentTuple in outputList:
        # Convert tuple to string:
        currentLine = str(currentTuple) + "\n"

        # Write to file:
        file.write(currentLine)

    # Close and return:
    file.close()
    print("Wrote out file: ", filePath)


# writeFiles(writeDir, listsToWrite, outFiles, overwriteFiles)
def writeDatasets(writeDir: str, listsToWrite: list, fileNames: list, fileExtension: str, overWriteFile: bool) -> None:
    """
    Write training and validation sample lists as text files:
    :param writeDir: Writing directory
    :param listsToWrite: Lists of samples of each dataset to write
    :param fileNames: text output filenames
    :param fileExtension: extension of the file (txt)
    :param overWriteFile: overwrite file flag
    :return: None
    """
    # Check if output dir exists:
    directoryExists = os.path.isdir(writeDir)
    if not directoryExists:
        raise ValueError("Error: Output directory does not exist. Dir: ", writeDir)

    # Write files:
    for i, currentList in enumerate(listsToWrite):
        # Prepare out text file:
        filePath = writeDir + fileNames[i] + fileExtension
        print("Writing out file: ", filePath)

        # Check if output dir exists:
        if os.path.exists(filePath):
            if not overWriteFile:
                print("File exists, skipping...")
                # If file already exists, skip iteration...
                continue

        # Write list in file:
        writeList2File(filePath, currentList)


def readListFromFile(filePath: str) -> list:
    """
    Open and parses a list of tuple strings from a text file
    :param filePath: path to the text file
    :return: list with tuples
    """
    # Open text file:
    file = open(filePath, "r")
    # Read file line by line
    # Returns each line as a string element in a list
    fileList = file.readlines()

    # Preprocess the list of tuples:
    outList = [ast.literal_eval(str(currentString)) for currentString in fileList]

    # Done:
    return outList


def createDataset(totalFiles: int) -> list:
    """
    Creates a dummy dataset, a list of strings representing filenames
    :param totalFiles: total files (strings) desired in the list
    :return: output list of filenames
    """
    datasetList = [("sample" + str(i * 2) + ".png", "sample" + str((i * 2) + 1) + ".png") for i in range(0, totalFiles)]
    return datasetList


# Output txt file names:
outFiles = ["trainSamples", "valSamples"]

# Dictionary of file names with item list (pair strings) and dataset name:
filesDict = {
    outFiles[0]: {"list": [], "name": "training", "newSamples": 0},
    outFiles[1]: {"list": [], "name": "validation", "newSamples": 0}
}

writeDir = "D://dataSets//faces//"

# Mode -> Read :  Reads dataset and READS the text files for validation and training
# Mode -> Write : Reads dataset and WRITES the text files for validation and training
mode = "read"

# Overwrite output files?
overwriteFiles = True

# Dataset partition:
trainPortion = 0.8
valPortion = 1.0 - trainPortion

# Set seed:
randomSeed = 42
random.seed(randomSeed)

# Dataset dictionary:
# Holds sample names and dataset types: {"sample01.png":"V"},
# Created using the info in the text files val and train samples:
datasetSamples = {}

if mode == "write":

    # Create the "Dataset" (2-item tuples of sample names):
    dataset = createDataset(totalFiles=20)

    # Shuffle the list of samples:
    random.shuffle(dataset)

    # Read the samples, create the two output files:
    print("Mode: Write...")

    # Get the number of total positive samples:
    totalSamples = len(dataset)

    print("Creating Train & Validation splits from " + str(totalSamples) + " samples...")

    # Create the partitions:
    trainSamples = math.ceil(trainPortion * totalSamples)

    # Split the dataset:
    trainList = dataset[0:trainSamples]
    validationList = dataset[trainSamples:]

    print("-> Train samples: ", str(len(trainList)))
    print("-> Validation samples: ", str(len(validationList)))

    # Write files:
    listsToWrite = [trainList, validationList]
    fileNames = [outFiles[0], outFiles[1]]
    writeDatasets(writeDir, listsToWrite, fileNames, ".txt", overwriteFiles)

# "Read" Mode:
elif mode == "read":

    # Create the "Dataset" (2-item tuples of sample names):
    dataset = createDataset(totalFiles=25)

    # Shuffle the list of samples:
    random.shuffle(dataset)

    print("Mode: Read...")

    # Read dataset files:
    pastSampleCount = {outFiles[0]: 0, outFiles[0]: 0}
    for currentName in outFiles:
        # Create file path:
        currentFile = writeDir + currentName + ".txt"

        # Check if path + file exist:
        fileExists = os.path.isfile(currentFile)

        if not fileExists:
            print("Error - File: ", currentFile, "does not exist.")
        else:
            print("Opening text file: ", currentFile)

            # Receive lists of filenames from text file, store it in
            # dictionary:
            filesDict[currentName]["list"] = readListFromFile(currentFile)
            print("Dataset - {" + str(filesDict[currentName]["name"]) + "} read successfully.")

            # Add to sample counter from files:
            pastSampleCount[currentName] = len(filesDict[currentName]["list"])

    # # Check for number of files mismatch:
    # currentTotalSamples = len(dataset)
    # if currentTotalSamples != fileSamples:
    #     print("[!] Warning - Stored datasets and read datasets do not have the same amount of samples!")
    #     print(" -> Datasets from files: ", fileSamples, " samples.")
    #     print(" -> Datasets from disk: ", currentTotalSamples, " samples.")
    #     input(">> Press Enter to continue...")

    # New samples are stored here:
    newSamples = []

    # # Process dataset
    # for currentName in outFiles:
    #     print("Processing dataset: ", filesDict[currentName]["name"])
    #     currentDataset = filesDict[currentName]["list"]

    # Loop through new dataset, get sample tuple:
    for currentSample in dataset:
        sampleFound = False
        # Check if current sample tuple is in either dataset:
        for currentName in outFiles:
            print("Checking in dataset: ", currentName)
            # Check both datasets:
            if currentSample in filesDict[currentName]["list"]:
                print("Sample found:", currentSample, "in:", currentName)
                # Found, set flag & break:
                sampleFound = True
                break

        # Sample not found:
        if not sampleFound:
            # This is a new sample:
            print("[New Sample]:", currentSample)
            newSamples.append(currentSample)

    # Print number of new samples:
    totalNewSamples = len(newSamples)
    print("Total New Samples: ", totalNewSamples)

    # Should the datasets be updated?
    if totalNewSamples != 0:
        print("Found new samples, updating datasets...")

        # Adjust dataset partitions:
        # Training samples are in filesDict[trainSamples]["list"]
        # Validation samples are in filesDict[valSamples]["list"]

        # New samples added and are yet to be spead across the lists
        # are in newSamples
        # New Dataset Size = Past Dataset (files read) + New Samples
        pastDatasetSize = pastSampleCount[outFiles[0]] + pastSampleCount[outFiles[1]]
        newDatasetSize = pastDatasetSize + totalNewSamples
        totalTrain = math.ceil(trainPortion * newDatasetSize)

        # Get number of new train samples and store it in dict:
        newTrainSamples = totalTrain - pastSampleCount[outFiles[0]]
        filesDict[outFiles[0]]["newSamples"] = newTrainSamples

        # Get number of new val samples and store it in dict:
        valNewSamples = totalNewSamples - newTrainSamples
        filesDict[outFiles[1]]["newSamples"] = valNewSamples

        # Check out the new numbers:
        for currentName in outFiles:
            print("New Samples in: " + str(currentName) + " ->", filesDict[currentName]["newSamples"])

        # Add new samples to training/validation lists:
        random.shuffle(newSamples)

        # Shallow copy of list before modifications:
        popSampleList = newSamples

        # Update datasets :
        newDatasetSize = 0
        # Add new samples to previous dataset splits:
        for currentName in outFiles:
            samplesToTake = filesDict[currentName]["newSamples"]

            # Get initial list length prior to extending:
            previousListLength = len(filesDict[currentName]["list"])

            # Check if there are new items to be added to the list:
            if samplesToTake != 0:

                # Get new elements from list:
                currentSlice = popSampleList[-samplesToTake:]

                # Pop elements from list:
                popSampleList = popSampleList[:-samplesToTake or None]
                # Add new sliced list to old list:
                filesDict[currentName]["list"] = (u := filesDict[currentName]["list"] + currentSlice)

            # Accumulate ds size:
            newDatasetSize += len(filesDict[currentName]["list"])

            print("[+] " + currentName, ": ", previousListLength, "->", len(filesDict[currentName]["list"]),
                  "[" + str(samplesToTake) + "]")

        # Check out the new dataset values:
        print("New dataset size:", newDatasetSize)
        print(" -> Training portion:", len(filesDict[outFiles[0]]["list"]) / newDatasetSize)
        print(" -> Validation portion:", len(filesDict[outFiles[1]]["list"]) / newDatasetSize)

        # List len = 0 check:
        if len(popSampleList) != 0:
            raise ValueError("New samples were not entirely assigned!. List still has: ", len(popSampleList),
                             "elements.")

        # Write new files:
        listsToWrite = [filesDict[outFiles[0]]["list"], filesDict[outFiles[1]]["list"]]
        fileNames = [outFiles[0], outFiles[1]]
        writeDatasets(writeDir, listsToWrite, fileNames, ".txt", True)

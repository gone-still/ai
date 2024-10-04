# File        :   DatasetRecorder.py
# Version     :   1.6.4
# Description :   Records the state of a dataset (samples in validation split +training split)

# Date:       :   Oct 3, 2024
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT

# To-do:
# Backup mechanism for past saved training and validation text files, so no
# overwriting occurs.

import os
import random
import math
import ast
from datetime import date, datetime
from typing import Union


# Module-level functions:
def writeList2File(filePath: str, outputList: list, verbose=False) -> None:
    """
    Writes a list of strings to a text file.

    :param filePath: path of the text file
    :param outputList: list of strings to write
    :param verbose: enable debug output
    :return: None
    """
    # Open file in write mode:
    file = open(filePath, "w")

    # Process tuples:
    for currentTuple in outputList:
        # Convert tuple to string:
        currentLine = str(currentTuple) + "\n"
        # print("Writing Line: ", currentLine)
        # Write to file:
        file.write(currentLine)

    # Close and return:
    file.close()
    if verbose:
        print("writeList2File>> Wrote out file: ", filePath)


def readListFromFile(filePath: str, verbose=False) -> list:
    """
    Open and parses a list of tuple strings from a text file.

    :param filePath: path to the text file
    :param verbose: enable debug output
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


def getDateString(verbose: bool = False) -> str:
    """
    Returns a string with the  current time + date
    :param verbose: debug flag
    :return: date string in the format: "_hh-mm-ss-Month-Day-Year"
    """
    # get date:
    currentDate = date.today()
    currentDate = currentDate.strftime("%b-%d-%Y")

    # Get time:
    currentTime = datetime.now()
    currentTime = currentTime.strftime("%H:%M:%S")

    # Drop them nasty ":s":
    timeString = "_" + currentTime[0:2] + "-" + currentTime[3:5] + "-" + currentTime[6:8]

    if verbose:
        print("getDateString>> Current Time: " + currentTime + " Date String: " + timeString)

    dateString = timeString + "-" + currentDate

    return dateString


class DatasetRecorder:

    # Initialization:
    def __init__(self, workingDirectory: str, trainSplit: float = 0.8, datasetNames=None,
                 fileExtension: str = ".txt", seed=42) -> None:
        """
        Initializes the dataset object.

        :param workingDirectory: the path of the working directory as string
        :param trainSplit: the portion of the dataset used for training (default 0.8)
        :param datasetNames: the dictionary of text file names as strings
        :param fileExtension: extension of the output file (default is ".txt")
        :param seed: an optional seed integer to feed the rng (default is 42)
        """

        # Set attributes:
        if datasetNames is None:
            datasetNames = {"training": "trainSamples", "validation": "valSamples"}

        self._workingDirectory = workingDirectory
        self._trainSplit = trainSplit
        self._datasetNames = datasetNames
        self._fileExtension = fileExtension
        self._verbose = False

        # Create the target directories:
        self._trainDirectory = os.path.join(self._workingDirectory,
                                            self._datasetNames["training"] + self._fileExtension)
        self._valDirectory = os.path.join(self._workingDirectory,
                                          self._datasetNames["validation"] + self._fileExtension)

        # Counters for each dataset partition:
        self._sampleCounters = {self._datasetNames["training"]: 0, self._datasetNames["validation"]: 0}

        # Dictionary of file names with item list (pair strings) and dataset name:
        self._filesDict = {
            self._datasetNames["training"]: {"list": [], "name": "training", "newSamples": 0},
            self._datasetNames["validation"]: {"list": [], "name": "validation", "newSamples": 0}
        }

        # Dictionary of datasets, holds each processed dataset for external use:
        self._datasetDict = {
            self._datasetNames["training"]: [],
            self._datasetNames["validation"]: [],
        }

        # Overwrite mode dictionary:
        self._overwriteModeTable = {"Overwrite": 1, "Keep": 2, "Backup": 3}

        # Set the random seed:
        self._seed = seed

        # Data leaks dictionary:
        self._dataLeaksDict = {"foundLeaks": 0, "leaks": [], "totalLeaks": 0, "totalTrainSamples": 0,
                               "totalValSamples": 0}

        print("DatasetRecorder>> [Training] Directory set to: ", self._trainDirectory)
        print("DatasetRecorder>> [Validation] Directory set to: ", self._valDirectory)

        # Paths to dataset names dictionary:
        self._pathsDictionary = {self._trainDirectory: self._datasetNames["training"],
                                 self._valDirectory: self._datasetNames["validation"]}

    def setVerbose(self, verbose=False):
        """
        Enables debug output.

        :param verbose: boolean flag to set the mode
        :return: None
        """
        self._verbose = verbose

        if self._verbose:
            print("DatasetRecorder>> Verbose mode enabled.")

    def createDataset(self, totalFiles: int, shuffleList=True) -> list:
        """
        Creates a dummy dataset, a list of strings representing filenames along a random binary label.

        :param totalFiles: total files (strings) desired in the list
        :return: output list of filenames
        """
        # Seed the rgn:
        random.seed(self._seed)
        # Create dataset:
        datasetList = [("sample" + str(i * 2) + ".png", "sample" + str((i * 2) + 1) + ".png", str(random.randint(0, 1)))
                       for i in range(0, totalFiles)]

        # Shuffle the list of samples:
        if shuffleList:
            random.shuffle(datasetList)

        return datasetList

    def writeDatasets(self, dataToWrite: list, overwriteCode: int = 1) -> None:
        """
        Write training and validation sample lists as text files.

        :param dataToWrite: Lists of tuples containing (sampleList, writePath)
        :param overwriteCode: overwrite mode as int (1 ->"Overwrite files", 2 -> "Do not overwrite",
        3 -> "Create backup before overwriting")
        :return: None
        """
        # Check if output dir exists:
        directoryExists = os.path.isdir(self._workingDirectory)
        if not directoryExists:
            raise ValueError("writeDatasets>> Error: Output directory does not exist. Dir: ", self._workingDirectory)

        # Data to write is received in the format:
        # [(trainList, trainPath), (valList, valPath)]

        # Write files:
        for i, currentTuple in enumerate(dataToWrite):
            # Extract list:
            currentList = currentTuple[0]
            # Extract path:
            currentPath = currentTuple[1]

            # Check if output file exists:
            if os.path.exists(currentPath):

                # Overwrite original files:
                if overwriteCode == 1:

                    if self._verbose:
                        print("writeDatasets>> [!] Found file: " + str(currentPath) + ". Overwriting.")

                    # Write list in file:
                    writeList2File(currentPath, currentList, self._verbose)

                # Do not overwrite files:
                elif overwriteCode == 2:

                    if self._verbose:
                        print("writeDatasets>> File: " + currentPath + " exists, skipping...")
                    # If file already exists, skip iteration...
                    continue

                else:

                    # "Back files before writing them" mode:
                    if self._verbose:
                        print("writeDatasets>> Backing up file...")

                    # Get date string:
                    dateString = getDateString(self._verbose)

                    # Get dataset name:
                    datasetName = self._pathsDictionary[currentPath]

                    # Append date string:
                    datasetName += dateString

                    # Create new path:
                    backupPath = os.path.join(self._workingDirectory, datasetName + self._fileExtension)

                    # Rename file:
                    os.rename(currentPath, backupPath)

                    if self._verbose:
                        print("writeDatasets>> Backed up file to:", backupPath)

                    # Write new file at original path:
                    writeList2File(currentPath, currentList, self._verbose)
            else:

                if self._verbose:
                    print("writeDatasets>> Creating File for the first time...")

                # Write list in file:
                writeList2File(currentPath, currentList, self._verbose)

    def saveDataset(self, currentDataset: Union[list, dict], overwriteMode="Overwrite") -> dict:
        """
        Method that writes a dataset in two files inside the working directory.
        Each file contains dataset samples spread according to the split portion indicated in the object constructor.

        :param currentDataset: can be list or dict. If list, pass all the dataset as one big list of paths and labels,
        the method will partition the dataset.
        If the dataset has been manually split, pass it as a dict of two lists with the keys "trainSamples" and "valSamples"
        for each partition.
        :param overwriteMode: overwrite mode as string, one of the following: "Overwrite", "Keep", "Backup"
        :return: A dictionary with each dataset in the keys provided by datasetNames.
        Default names are "trainSamples" and "valSamples".
        """

        # Get dataset type> list -> create partitions, dict -> use manual partitions
        datasetType = type(currentDataset)

        if datasetType is list:

            if self._verbose:
                print("saveDataset>> Splitting dataset in training and validation portions...")

            # Get the number of total positive samples:
            totalSamples = len(currentDataset)

            # Check if list is empty
            if totalSamples == 0:
                raise ValueError("saveDataset>> Got an empty dataset list.")

            # Read the samples, create the two output files:
            if self._verbose:
                print("saveDataset>> Creating Train & Validation splits from " + str(totalSamples) + " samples...")

            # Create the partitions:
            trainSamples = math.ceil(self._trainSplit * totalSamples)

            # Split the dataset:
            trainList = currentDataset[0:trainSamples]
            validationList = currentDataset[trainSamples:]

        elif datasetType is dict:

            # Use manual partition:
            totalDatasets = len(currentDataset)

            # Check if we have exactly 2 datasets:
            if totalDatasets != 2:
                raise ValueError("saveDataset>> Expected 2 keys {Train/Valid} in dataset dictionary, got: ",
                                 totalDatasets)

            # Check keys:
            for key in [self._datasetNames["training"], self._datasetNames["validation"]]:
                if key not in currentDataset:
                    raise ValueError("saveDataset>> Key: " + str(key) + " not found in dataset.")

            # Store partitions:
            trainList = currentDataset[self._datasetNames["training"]]
            validationList = currentDataset[self._datasetNames["validation"]]

        else:

            # Type not supported:
            raise ValueError("saveDataset>> Dataset type not supported. Got: ", datasetType, ". Expected list or dict.")

        if self._verbose:
            print(" -> Train samples: ", str(len(trainList)))
            print(" -> Validation samples: ", str(len(validationList)))

        # Data to write:
        listsToWrite = [(trainList, self._trainDirectory), (validationList, self._valDirectory)]

        # Check overwrite mode:
        if overwriteMode in self._overwriteModeTable:

            # Get overwrite code and call write dataset function:
            overwriteCode = self._overwriteModeTable[overwriteMode]
            self.writeDatasets(listsToWrite, overwriteCode)

        else:
            # Mode not supported:
            raise ValueError("saveDataset>> Error. Overwrite mode not supported. Options are: ",
                             self._overwriteModeTable)

        # Pack the results in dict:
        self._datasetDict[self._datasetNames["training"]] = trainList
        self._datasetDict[self._datasetNames["validation"]] = validationList

        # Return the processed datasets:
        return self._datasetDict

    def updateDataset(self, currentDataset: list, overwriteFiles=True) -> dict:
        """
        Method that loads a dataset as two files and updates them with new samples according to the
        Train/Validation split.

        :param currentDataset: The new, input dataset that contains the last saved samples + new samples
        :param overwriteFiles: Flag to overwrite the target text files
        :return: A dictionary with each dataset in the keys provided by datasetNames.
        Default names are "trainSamples" and "valSamples".
        """

        # List with output file names and target directories:
        datasetInfo = [(self._datasetNames["training"], self._trainDirectory),
                       (self._datasetNames["validation"], self._valDirectory)]

        # Read dataset files:
        for currentTuple in datasetInfo:

            # Get current dataset name:
            currentName = currentTuple[0]

            # Get the target file path:
            currentFile = currentTuple[1]

            # Check if file exists:
            fileExists = os.path.isfile(currentFile)

            if not fileExists:
                raise ValueError("updateDataset>> Error: File", currentFile, " does not exist.")
            else:

                if self._verbose:
                    print("updateDataset>> Opening text file: ", currentFile)

                # Receive lists of filenames/samples from text file, store it in dictionary:
                self._filesDict[currentName]["list"] = readListFromFile(currentFile, self._verbose)

                if self._verbose:
                    print("Dataset - {" + str(self._filesDict[currentName]["name"]) + "} read successfully.")

                # Add to sample counter from files, this is the amount of samples per dataset that is stored
                # in each file. Gotta keep track of this to calculate how many new samples will be added and how
                # the split portions get adjusted:
                self._sampleCounters[currentName] = len(self._filesDict[currentName]["list"])

        # New samples are stored here:
        newSamples = []

        # Loop through the new dataset, get sample tuple:
        for currentSample in currentDataset:
            sampleFound = False

            # Check if current sample tuple is in either dataset:
            for currentTuple in datasetInfo:

                # Get current dataset name:
                currentName = currentTuple[0]

                if self._verbose:
                    print("updateDataset>> Checking sample in dataset: ", currentName)

                # Check both datasets:
                if currentSample in self._filesDict[currentName]["list"]:
                    if self._verbose:
                        print("updateDataset>> Sample found:", currentSample, "in:", currentName)
                    # Found, set flag & break:
                    sampleFound = True
                    break

            # Sample not found:
            if not sampleFound:

                # This is a new sample:
                newSamples.append(currentSample)

                if self._verbose:
                    print(" [New Sample]:", currentSample)

        # Print number of new samples:
        totalNewSamples = len(newSamples)

        if self._verbose:
            print("updateDataset>> Total New Samples: ", totalNewSamples)

        # Should the datasets be updated?
        if totalNewSamples != 0:

            if self._verbose:
                print("updateDataset>> Found new samples, updating datasets...")

            # New samples added and are yet to be spead across the lists are currently stored in "newSamples"
            # New Dataset Size = Past Dataset (files read) + New Samples
            pastDatasetSize = self._sampleCounters[self._datasetNames["training"]] + self._sampleCounters[
                self._datasetNames["validation"]]
            newDatasetSize = pastDatasetSize + totalNewSamples
            totalTrain = math.ceil(self._trainSplit * newDatasetSize)

            # Get number of new train samples and store it in dict:
            newTrainSamples = totalTrain - self._sampleCounters[self._datasetNames["training"]]
            self._filesDict[self._datasetNames["training"]]["newSamples"] = newTrainSamples

            # Get number of new val samples and store it in dict:
            valNewSamples = totalNewSamples - newTrainSamples
            self._filesDict[self._datasetNames["validation"]]["newSamples"] = valNewSamples

            # Check out the new numbers:
            if self._verbose:
                for currentTuple in datasetInfo:
                    # Get current dataset name:
                    currentName = currentTuple[0]
                    print("New Samples in: " + str(currentName) + " ->", self._filesDict[currentName]["newSamples"])

            # Add new samples to training/validation lists:
            random.seed(self._seed)
            random.shuffle(newSamples)

            # Shallow copy of list before modifications:
            popSampleList = newSamples

            # Update datasets :
            newDatasetSize = 0
            # Add new samples to previous dataset splits:
            for currentTuple in datasetInfo:

                # Get current dataset name:
                currentName = currentTuple[0]

                samplesToTake = self._filesDict[currentName]["newSamples"]

                # Get initial list length prior to extending:
                previousListLength = len(self._filesDict[currentName]["list"])

                # Check if there are new items to be added to the list:
                if samplesToTake != 0:
                    # Get new elements from list:
                    currentSlice = popSampleList[-samplesToTake:]

                    # Pop elements from list:
                    popSampleList = popSampleList[:-samplesToTake or None]
                    # Add new sliced list to old list:
                    self._filesDict[currentName]["list"] = (u := self._filesDict[currentName]["list"] + currentSlice)

                # Accumulate ds size:
                newDatasetSize += len(self._filesDict[currentName]["list"])

                if self._verbose:
                    print("[+] " + currentName, ": ", previousListLength, "->",
                          len(self._filesDict[currentName]["list"]),
                          "[" + str(samplesToTake) + "]")

            # Check out the new dataset values:
            if self._verbose:
                print("New dataset size:", newDatasetSize)
                print(" -> Training portion:",
                      len(self._filesDict[self._datasetNames["training"]]["list"]) / newDatasetSize)
                print(" -> Validation portion:",
                      len(self._filesDict[self._datasetNames["validation"]]["list"]) / newDatasetSize)

            # List len = 0 check:
            if len(popSampleList) != 0:
                raise ValueError("New samples were not entirely assigned!. List still has: ", len(popSampleList),
                                 "elements.")

            # Write new files,
            # Call the list containing the datasets:
            trainList = self._datasetDict[self._datasetNames["training"]] = \
                self._filesDict[self._datasetNames["training"]]["list"]
            validationList = self._datasetDict[self._datasetNames["validation"]] = \
                self._filesDict[self._datasetNames["validation"]]["list"]

            # Pack the lists into a list of two elements:
            listsToWrite = [(trainList, self._trainDirectory), (validationList, self._valDirectory)]

            # Actually write the data at their path:
            self.writeDatasets(listsToWrite, overwriteFiles)

            # Pack the results in dict:
            self._datasetDict[self._datasetNames["training"]] = trainList
            self._datasetDict[self._datasetNames["validation"]] = validationList

        else:
            print("updateDataset>> No new samples found. Skipping update process.")

        # Return the processed datasets:
        return self._datasetDict

    def checkDataLeaks(self, pathList: list = None) -> dict:
        """
        Function that checks two files (given as string paths) for data leakages

        :param pathList: a list with two strings denoting the target paths, if None,
        the configured (default) paths are used instead
        :return: Dictionary with keys:
        foundLeaks - Found leaks bool,
        leaks - The list of duplicated items,
        totalLeaks - Total of leaks found as integer
        totalTrainSamples - Total of train samples in the full dataset
        totalValSamples - Total of validation samples in the full dataset
        """

        # Set the file paths:
        if pathList is None:
            # Set default file paths:
            filePaths = [self._trainDirectory, self._valDirectory]
        else:
            # Check list length. Needs two elements:
            if len(pathList) != 2:
                raise ValueError("checkDataLeaks>> Filename list got: ", len(pathList), "items but expected 2.")
            # Use provided paths:
            filePaths = pathList

        # Check if files exist:
        for currentFile in filePaths:
            fileExists = os.path.isfile(currentFile)
            if not fileExists:
                raise ValueError("checkDataLeaks>> Error: File", currentFile, " does not exist.")

        # Open files into lists:
        datasets = [readListFromFile(currentFile) for currentFile in filePaths]

        # Get number of samples:
        trainSamples = len(datasets[0])
        valSamples = len(datasets[1])

        # Check intersection between the two lists:
        intersectionSet = list(set(datasets[0]).intersection(datasets[1]))
        # How many elements:
        nIntersections = len(intersectionSet)

        foundDuplicates = False
        if nIntersections != 0:
            foundDuplicates = True

        # Pack results:
        self._dataLeaksDict["foundLeaks"] = foundDuplicates
        self._dataLeaksDict["leaks"] = intersectionSet
        self._dataLeaksDict["totalLeaks"] = nIntersections

        self._dataLeaksDict["totalTrainSamples"] = trainSamples
        self._dataLeaksDict["totalValSamples"] = valSamples

        # Done:
        return self._dataLeaksDict

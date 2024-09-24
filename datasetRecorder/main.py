# File        :   main.py
# Version     :   1.1.0
# Description :   Usage example of the DatasetRecorder module.
#                 The module records and resumes the "state" of a dataset as a pair of txt files while
#                 avoiding data leaks and maintaining a training/validation split.

# Date:       :   Sept 24, 2024
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT

import os
from DatasetRecorder import DatasetRecorder


def printDataset(inputDict: dict) -> None:
    """
    Just prints the content of a dictionary dataset
    :param inputDict: the dict dataset to print
    :return: None
    """
    for dictTuple in inputDict.items():
        print("-> Dataset: ", dictTuple[0])
        print("   Samples:", dictTuple[1])


# Set the working directory. The working directory will be the target from/to
# data files are written and read
writeDir = "D://dataSets//faces//"

# The name of the dataset files are, by default:
# Training dataset -> trainSamples (will produce the text file - trainSamples.txt)
# Validation dataset -> valSamples (will produce the text file - valSamples.txt)

# Create the object:
# The default split for training and validation is 80/20
myDatasetRecorder = DatasetRecorder(writeDir, 0.8)

# Set verbose:
myDatasetRecorder.setVerbose(True)

# Create a dummy dataset:
# The method will create a list of tuples with dummy filenames and a random binary class,
# The format of the output is:
# [("sample1.png", "sample2.png", "0"), ("sample3.png", "sample4.png", "1"), ...]
# The list items will be shuffled by default
dummyDataset = myDatasetRecorder.createDataset(totalFiles=20)

# Write (save) the dataset:
# The method will partition the dataset in two splits: Training and Validation
# according to the default split portion and will produce two text files
# inside the working directory.
# "Backup" mode preserves original files IF they are found.
# The method also returns the processed dataset as a dictionary:
myDataset = myDatasetRecorder.saveDataset(dummyDataset, overwriteMode="Backup")

# print the dataset:
printDataset(myDataset)

# Let's create another dataset, this time with some more samples (25 new samples, in fact).
# This is supposed to emulate the gathering of new samples after the original dataset was used
# and saved by the DatasetRecorder object:
dummyDataset = myDatasetRecorder.createDataset(totalFiles=45)

# Read and amend (update) the previous dataset state:
# The method will read the dataset's past state and will append the new samples
# while keeping the original split.
# It also returns the updated dataset as a dictionary:
myDataset = myDatasetRecorder.updateDataset(dummyDataset)

# print the dataset:
printDataset(myDataset)

# Check for data leakages, the method returns a dict with results:
# The argument to this function can be a list containing the paths to the datasets,
# if no argument is provided the default paths are used instead:
leakResults = myDatasetRecorder.checkDataLeaks()

# Flag is in the "foundLeaks" key:
foundLeaks = leakResults["foundLeaks"]
print("Found Leaks?", foundLeaks)

# Print results:
print("Leak Results:", leakResults)

# Show some other stats:
totalTrainSamples = leakResults["totalTrainSamples"]
totalValSamples = leakResults["totalValSamples"]

totalDatasetSamples = totalTrainSamples + totalValSamples
print("Training Portion: ", totalTrainSamples / totalDatasetSamples,
      "(" + str(totalTrainSamples) + "/" + str(totalDatasetSamples) + ")")
print("Validation Portion: ", totalValSamples / totalDatasetSamples,
      "(" + str(totalValSamples) + "/" + str(totalDatasetSamples) + ")")

# File        :   mnistWritter.py
# Version     :   1.0.0
# Description :   Writes samples of the mnist numerical dataset
#                 as png images per class (test & train)

# Date:       :   Jun 26, 2023
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT

import cv2
import os
import numpy as np
from tensorflow.keras.datasets import mnist


# Defines a re-sizable image window:
def showImage(imageName, inputImage):
    cv2.namedWindow(imageName, cv2.WINDOW_NORMAL)
    cv2.imshow(imageName, inputImage)
    cv2.waitKey(0)


# Writes an PGN image:
def writeImage(imagePath, inputImage):
    cv2.imwrite(imagePath, inputImage, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print("Wrote Image: " + imagePath)


projectPath = "D://dataSets//faces//mnist//"
filenamePrefix = "image_"
imageExtension = ".png"

displayImages = True

totalClasses = 10
datasetNames = ["train", "test"]
samplesPerClass = {"train": 60, "test": 10}

# Load the dataset:
(trainX, trainY), (testX, testY) = mnist.load_data()

totalImages = len(trainX)
totalLabels = len(trainY)

print("[INFO] Number of Images: " + str(totalImages))
print("[INFO] Number of Labels: " + str(totalLabels))

# Create the "dictionary of classes":
classesDictionary = {str(i): i for i in range(totalClasses)}

# Counter dict for each dataset:
counterDict = {"train": {}, "test": {}}
samplesReady = {"train": [0] * totalClasses, "test": [0] * totalClasses}
imageIndices = {"train": 0, "test": 0}
writeCounters = {"train": 0, "test": 0}

writeSamples = True
while writeSamples:

    # For each dataset:
    for currentDataset in datasetNames:

        # Set image source:
        if currentDataset == "train":
            imageSource = trainX
            labelSource = trainY
        else:
            imageSource = testX
            labelSource = testY

        # Get image index:
        imageIndex = imageIndices[currentDataset]

        # Get the image:
        currentImage = imageSource[imageIndex]
        # Get the class:
        currentClass = str(labelSource[imageIndex])

        # Get counter:
        if currentClass not in counterDict[currentDataset]:
            counterDict[currentDataset][currentClass] = 0

        currentCounter = counterDict[currentDataset][currentClass]

        # Should we process this sample? (max -> samplesPerClass):
        if currentCounter < samplesPerClass[currentDataset]:
            # Add "color" dimension:
            currentImage = np.expand_dims(currentImage, axis=-1)
            # To BGR:
            currentImage = cv2.cvtColor(currentImage, cv2.COLOR_GRAY2BGR)

            # Show image
            if displayImages:
                showImage("Current Image", currentImage)

            # Output directory:
            outDirectory = projectPath + currentDataset + "//" + currentClass

            # Create output directory with first processed image
            # from this class:
            if currentCounter == 0:

                print("Checking directory: " + outDirectory)
                directoryExists = os.path.isdir(outDirectory)

                if not directoryExists:
                    print("Creating Directory: " + outDirectory)
                    os.mkdir(outDirectory)
                    directoryExists = os.path.isdir(outDirectory)
                    if directoryExists:
                        print("Successfully created directory: " + outDirectory)
                else:
                    print("Directory Found.")

            # Write image:
            imagePath = outDirectory + "//" + filenamePrefix + str(currentCounter) + ".png"
            writeImage(imagePath, currentImage)

            # Sample counter goes up:
            counterDict[currentDataset][currentClass] += 1
            # Write counter goes up:
            writeCounters[currentDataset] += 1

            print("[" + currentDataset + "]", currentClass, "filled with: ", counterDict[currentDataset][currentClass],
                  "samples")

        else:

            # Class is filled with samples...
            idx = int(currentClass)
            # Same class in the train and test datasets should add up to 1
            # if all samples have been collected for both datasets:
            samplesReady[currentDataset][idx] = 0.5

            print("[" + currentDataset + "]", currentClass, "filled with: ", counterDict[currentDataset][currentClass],
                  "samples", "overall:", samplesReady[currentDataset][idx])

        # Next sample...
        imageIndices[currentDataset] += 1

        # Print totals:
        print("Train Total:", writeCounters["train"], "Test Total:", writeCounters["test"])

    # Is the dataset filled?
    datasetChecksum = 0
    for datasetName in datasetNames:
        datasetChecksum += sum(samplesReady[datasetName])

    # The "checksum" should be equal to the total number of classes:
    if datasetChecksum == totalClasses:
        writeSamples = False

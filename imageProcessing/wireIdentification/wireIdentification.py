# File        :   wireIdentification.py
# Version     :   1.0.0
# Description :   Script that identifies (in order) the colors of a set of electrical wires

# Date:       :   Aug 21, 2024
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT

import numpy as np
import cv2

import matplotlib.pyplot as plt


def readImage(imagePath):
    """Reads image via OpenCV"""
    # Loads image:
    inputImage = cv2.imread(imagePath)
    # Checks if image was successfully loaded:
    if inputImage is None:
        raise TypeError("readImage>> Error: Could not load Input image.")
    return inputImage


def showImage(imageName, inputImage):
    """Shows an image in a window"""
    cv2.namedWindow(imageName, cv2.WINDOW_NORMAL)
    cv2.imshow(imageName, inputImage)
    cv2.waitKey(0)


def getIntensityPeaks(y):
    """Gets local minima and maxima in an input array"""
    # Get function peaks:

    functionPeaks = np.where((y[2:-1] > y[1:-2]) * (y[2:-1] > y[3:]))[0] + 1
    # dips = np.where((y[1:-1] < y[0:-2]) * (y[1:-1] < y[2:]))[0] + 1

    return functionPeaks


def plotFunctionPeaks(inputFunction, inputPeaks, showPlot=True):
    """PLots a function and its local maxima"""
    x = range(len(inputFunction))
    plt.plot(x, inputFunction, color="red")
    plt.plot(inputPeaks, inputFunction[inputPeaks], "x", color="blue")

    if showPlot:
        plt.show()


# HSV Dictionary with known color min and max ranges:
hsvDictionary = {"Purple": {"max": [139, 255, 169], "min": [119, 124, 109]},
                 "Blue": {"max": [123, 255, 169], "min": [90, 179, 109]},
                 "Green": {"max": [86, 255, 169], "min": [71, 179, 109]},
                 "Yellow": {"max": [51, 227, 255], "min": [17, 120, 156]},
                 "Orange": {"max": [11, 255, 255], "min": [6, 120, 159]},
                 "Brown": {"max": [80, 135, 213], "min": [0, 83, 85]},
                 "Red": {"max": [177, 255, 255], "min": [155, 213, 166]},
                 "Gray": {"max": [151, 66, 192], "min": [137, 19, 165]},
                 "White": {"max": [165, 20, 237], "min": [118, 0, 160]},
                 "Black": {"max": [154, 185, 59], "min": [78, 0, 37]}}

# Set image path
imagePath = "D://opencvImages//colorWires.png"

# Load image:
inputImage = readImage(imagePath)
showImage("Input Image", inputImage)

# To grayscale:
greyImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
showImage("greyImage", greyImage)

# Apply Gaussian blur:
gaussianBlur = cv2.GaussianBlur(greyImage, (15, 15), 0)

# Get reduced array:
newReduced = cv2.reduce(gaussianBlur, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)

# Get image dims:
imageHeight, imageWidth = newReduced.shape[0:2]

# Get function peaks:
peaks = getIntensityPeaks(newReduced[0])

print("Peaks found: ", len(peaks))

# Plot function and peaks:
plotFunctionPeaks(newReduced[0], peaks, True)

# Check out sampling points:
# Peak position mask array:
maskArray = np.zeros(imageWidth, dtype=np.uint8)

# These are the locations of the peaks:
maskArray[peaks] = 1

# Create sampling point BGR image for visualization:
samplingPoints = np.zeros((imageHeight, imageWidth, 3), np.uint8)

# Split channels
b, g, r = cv2.split(inputImage)

# Store reduced channels here:
channelsList = []

for currentChannel in [b, g, r]:
    # Reduce matrix to a n row x 1 columns matrix:
    reducedImage = cv2.reduce(currentChannel, 0, cv2.REDUCE_MAX)
    # Into the reduced channels list:
    channelsList.append(reducedImage)

    showImage("Reduced", reducedImage)

# Merge:
mergedImage = cv2.merge(channelsList)
showImage("Merged", mergedImage)

# Get image dims:
imageHeight, imageWidth = mergedImage.shape[0:2]

# Use mask array to place pixel in sampling point BGR image:
for channel, color in enumerate([255, 0, 255]):
    # Checks location in mask array, if the value is 1, it places a channel from the input pixel
    # (255,0,255) -> hot pink in the current channel image. "Where" only handles 1-channel images:
    samplingPoints[:, :, channel] = np.where(maskArray == 1, color, mergedImage[:, :, channel])

# Colors will be sampled on the location shown in pink:
showImage("Sampling Points", samplingPoints)

# Sample pixels:
sampledPixels = []
# Check the mask/location array:
for i, getValue in enumerate(maskArray):
    # Check if this position is a sample position:
    if getValue == 1:
        # Get pixel from reduced BGR image:
        currentPixel = mergedImage[0][i]
        # Into the sampled colors list:
        sampledPixels.append(currentPixel)

print("Total pixels sampled: ", len(sampledPixels))

# Store unique colors here:
colorList = []

# Run every sampled color against the HSV dictionary to see if the sample color matches
# a valid dictionary color:
for i, currentPixel in enumerate(sampledPixels):
    colorFound = ""
    pixelLocation = -1

    # Create BGR Mat:
    pixelMat = np.zeros([1, 1, 3], dtype=np.uint8)
    pixelMat[0, 0] = currentPixel

    # Convert BGR "mat" to HSV:
    hsvPixel = cv2.cvtColor(pixelMat, cv2.COLOR_BGR2HSV)

    # Get Color data:
    for colorName in hsvDictionary:
        # Get data from dict:
        hsvData = hsvDictionary[colorName]

        # Get range data:
        maxRange = hsvData["max"]
        minRange = hsvData["min"]

        # Get color mask:
        # By thresholding the hsv pixel at this range, the mask will be white if the
        # color is found in the min-max range:
        colorMask = cv2.inRange(hsvPixel, np.array(minRange), np.array(maxRange))
        showImage("Color Mask", colorMask)

        # Mask must be entirely white:
        isWhite = cv2.countNonZero(colorMask)

        if isWhite == 1:
            # Get the color name:
            colorFound = colorName
            # Get peak location:
            pixelLocation = peaks[i]

            # Draw a point in the target pixel location:
            cv2.line(mergedImage, (pixelLocation, 0), (pixelLocation, 0), (0, 0, 255), 1)
            showImage("Found Color", mergedImage)

    # Let's see if the color was found in the hsv dictionary:
    if colorFound == "":
        # Unknown color:
        print("Color not found. Color:", i)
        colorString = "[?]"
    else:
        print("Seems like: ", colorFound, "at:", pixelLocation)
        # add to list:
        colorString = colorFound

    # Avoid consecutive color names:
    if len(colorList) > 0:
        # Get last string:
        lastString = colorList[-1]
        # Check different:
        if lastString != colorString:
            # Add to list:
            colorList.append(colorString)
    else:
        # Add to list:
        colorList.append(colorString)

print("Colors found (ordered):", colorList)

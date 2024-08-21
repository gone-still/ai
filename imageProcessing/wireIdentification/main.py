import numpy as np
import cv2
import math
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


# Set image path
imagePath = "D://opencvImages//colorWires.png"

# Load image:
inputImage = readImage(imagePath)
showImage("Input Image", inputImage)

# Split
b, g, r = cv2.split(inputImage)
reducedList = []

outPixel = np.zeros((1, 1), dtype=np.uint8)

for currentChannel in [b, g, r]:
    # Reduce matrix to a n row x 1 columns matrix:
    reducedImage = cv2.reduce(currentChannel, 0, cv2.REDUCE_MIN)
    reducedList.append(reducedImage)

    showImage("Reduced", reducedImage)

# Merge:
mergedImage = cv2.merge(reducedList)
showImage("Merged", mergedImage)

# Get image dims:
imageHeight, imageWidth = mergedImage.shape[0:2]

# To grayscale:
greyImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
showImage("greyImage", greyImage)

# Apply Gaussian blur:
gaussianBlur = cv2.GaussianBlur(greyImage, (15, 15), 0)

# Get reduced array:
newReduced = cv2.reduce(gaussianBlur, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)

# Get function peaks:
y = newReduced[0]

peaks = np.where((y[2:-1] > y[1:-2]) * (y[2:-1] > y[3:]))[0] + 1
dips = np.where((y[1:-1] < y[0:-2]) * (y[1:-1] < y[2:]))[0] + 1

print("Peaks found: ", len(peaks))

# Plot function and peaks:
x_values = range(len(newReduced[0]))
plt.plot(x_values, newReduced[0], color="red")
plt.plot(peaks, y[peaks], "x", color="blue")
# plt.plot(dips, y[dips], "x", color="orange")

# plt.show()

# Check out sampling points:
# Peak position mask array:
maskArray = np.zeros(imageWidth, dtype=np.uint8)
# These are the locations of the peaks:
maskArray[peaks] = 1

# Create sampling point BGR image for visualization:
samplingPoints = np.zeros((imageHeight, imageWidth, 3), np.uint8)

# Use mask array to place pixel in sampling point BGR image:
for channel, color in enumerate([255, 0, 255]):
    samplingPoints[:, :, channel] = np.where(maskArray == 1, color, mergedImage[:, :, channel])

showImage("Sampling Points", samplingPoints)

# Sample pixels:
sampledPixels = []
for i, getValue in enumerate(maskArray):
    if getValue == 1:
        currentPixel = mergedImage[0][i]
        sampledPixels.append(currentPixel)

print("Total pixels sampled: ", len(sampledPixels))

# Purple 	(hMin = 119 , sMin = 124, vMin = 109), (hMax = 139 , sMax = 255, vMax = 169)
# Blue 	(hMin = 90 , sMin = 179, vMin = 109), (hMax = 123 , sMax = 255, vMax = 169)
# Green 	(hMin = 71 , sMin = 179, vMin = 109), (hMax = 86 , sMax = 255, vMax = 169)
# Yellow 	(hMin = 17 , sMin = 120, vMin = 156), (hMax = 51 , sMax = 227, vMax = 255)
# Orange 	(hMin = 6 , sMin = 120, vMin = 159), (hMax = 11 , sMax = 255, vMax = 255)
# Brown 	(hMin = 0 , sMin = 83, vMin = 85), (hMax = 80 , sMax = 135, vMax = 213)
# Red 	(hMin = 155 , sMin = 213, vMin = 166), (hMax = 177 , sMax = 255, vMax = 255)
# Gray 	(hMin = 137 , sMin = 19, vMin = 165), (hMax = 151 , sMax = 66, vMax = 192)
# White 	(hMin = 118 , sMin = 0, vMin = 160), (hMax = 165 , sMax = 20, vMax = 237)
# Black 	(hMin = 78 , sMin = 0, vMin = 37), (hMax = 154 , sMax = 185, vMax = 59)

hsvDictionary = {"Purple": {"max": [139, 255, 169], "min": [119, 124, 109]},
                 "Blue": {"max": [123, 255, 169], "min": [90, 179, 109]}}

colorSet = set()

for i, currentPixel in enumerate(sampledPixels):
    colorFound = ""
    pixelLocation = -1

    # Create BGR Mat:
    pixelMat = np.zeros([1, 1, 3], dtype=np.uint8)
    pixelMat[0, 0] = currentPixel

    # Convert BGR "mat" to HSV:
    hsvPixel = cv2.cvtColor(pixelMat, cv2.COLOR_BGR2HSV)

    targetPixel = hsvPixel[0][0]

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

        for c in range(0, 3):
            maxValue = hsvData["max"][c]
            minValue = hsvData["min"][c]

            testValue = targetPixel[c]

            if minValue <= testValue < maxValue:
                if c == 2:
                    colorFound = colorName
                    pixelLocation = peaks[i]
                    # Draw a point in the target pixel location:
                    cv2.line(mergedImage, (pixelLocation, 0), (pixelLocation, 0), (0, 0, 255), 1)
                    showImage("Found Color", mergedImage)
            else:
                break

    # Let's see if the color was found in the hsv dictionary:
    if colorFound == "":
        print("Color not found. Color:", i)
    else:
        print("Seems like: ", colorFound, "at:", pixelLocation)
        # add to set:
        colorSet.add(colorFound)


print("Colors found (ordered):", colorSet)



# Automatic Object Detection based on HSV-sampling
# Answer for: https://stackoverflow.com/questions/67390790/object-detection-with-opencv-python/67393490#67393490
# Original Author: State Machine

# Imports:
import cv2
import numpy as np


# Defines a re-sizable image window:
def showImage(imageName, inputImage):
    cv2.namedWindow(imageName, flags=cv2.WINDOW_GUI_NORMAL)
    cv2.imshow(imageName, inputImage)
    cv2.waitKey(0)


# image path
fileName = "4dzfr.png"

# Reading an image in default mode:
inputImage = cv2.imread(fileName)

# Check image loading:
if inputImage is None:
    print("Could not load Input image.")

# Deep copy for final results:
inputImageCopy = inputImage.copy()

# Convert RGB to grayscale:
grayscaleImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

# Get binary image via Otsu:
binaryImage = np.where(grayscaleImage >= 200, 255, 0)
# The above operation converted the image to 32-bit float,
# convert back to 8-bit uint
binaryImage = binaryImage.astype(np.uint8)
# Show the image:
showImage("Binary Image", binaryImage)

# Find the contours on the binary image:
contours, hierarchy = cv2.findContours(binaryImage, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# Store the sampled pixels here:
sampledPixels = []

# Deep copy for temp results:
tempCopy = inputImage.copy()

# Look for the outer bounding boxes (no children):
for i, c in enumerate(contours):

    # Get the contour bounding rectangle:
    boundRect = cv2.boundingRect(c)

    # Get the dimensions of the bounding rect:
    rectX = boundRect[0]
    rectY = boundRect[1]
    rectWidth = boundRect[2]
    rectHeight = boundRect[3]

    # Compute the aspect ratio:
    aspectRatio = rectWidth / rectHeight

    # Create the filtering threshold value:
    delta = abs(0.7 - aspectRatio)
    epsilon = 0.1

    # Get the hierarchy:
    currentHierarchy = hierarchy[0][i][3]

    # Prepare the list of sampling points (One for the ellipse, one for the circle):
    samplingPoints = [(rectX - rectWidth, rectY), (rectX, rectY - rectHeight)]

    # Look for the target contours:
    if delta < epsilon and currentHierarchy == -1:

        # This list will hold both sampling pixels:
        pixelList = []

        # Get sampling pixels from the two locations:
        for s in range(2):
            # Get sampling point:
            sampleX = samplingPoints[s][0]
            sampleY = samplingPoints[s][1]

            # Get sample BGR pixel:
            samplePixel = inputImageCopy[sampleY, sampleX]

            # Store into temp list:
            pixelList.append(samplePixel)

        # convert list to tuple:
        pixelList = tuple(pixelList)

        # Save pixel value:
        sampledPixels.append(pixelList)

        # Draw the rectangle numbers:
        color = (0, 0, 255)
        cv2.rectangle(tempCopy, (int(rectX), int(rectY)),
                      (int(rectX + rectWidth), int(rectY + rectHeight)), color, 2)

        # Show the sampling points:
        showImage("Sampling Points", tempCopy)

# Store the bounding rectangles here:
boundingRectangles = []

# Loop through sampled pixels:
for p in range(len(sampledPixels)):
    # Get current pixel tuple:
    currentPixelTuple = sampledPixels[p]

    # Prepare the HSV mask:
    imageHeight, imageWidth = binaryImage.shape[:2]
    hsvMask = np.zeros((imageHeight, imageWidth), np.uint8)

    # Process the two sampling pixels:
    for m in range(len(currentPixelTuple)):
        # Get current pixel in the list:
        currentPixel = currentPixelTuple[m]

        # Create BGR Mat:
        pixelMat = np.zeros([1, 1, 3], dtype=np.uint8)
        pixelMat[0, 0] = currentPixel

        # Convert the BGR pixel to HSV:
        hsvPixel = cv2.cvtColor(pixelMat, cv2.COLOR_BGR2HSV)
        H = hsvPixel[0][0][0]
        S = hsvPixel[0][0][1]
        V = hsvPixel[0][0][2]

        # Create HSV range for this pixel:
        rangeThreshold = 5
        lowerValues = np.array([H - rangeThreshold, S - rangeThreshold, V - rangeThreshold])
        upperValues = np.array([H + rangeThreshold, S + rangeThreshold, V + rangeThreshold])

        # Create HSV mask:
        hsvImage = cv2.cvtColor(inputImage.copy(), cv2.COLOR_BGR2HSV)
        tempMask = cv2.inRange(hsvImage, lowerValues, upperValues)
        hsvMask = hsvMask + tempMask

    # Set kernel (structuring element) size:
    kernelSize = 3
    # Set morph operation iterations:
    opIterations = 2
    # Get the structuring element:
    morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))
    # Perform closing:
    hsvMask = cv2.morphologyEx(hsvMask, cv2.MORPH_CLOSE, morphKernel, None, None, opIterations, cv2.BORDER_REFLECT101)

    # Process current contour:
    currentContour, _ = cv2.findContours(hsvMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for _, c in enumerate(currentContour):
        # Get the contour's bounding rectangle:
        boundRect = cv2.boundingRect(c)

        # Get the dimensions of the bounding rect:
        rectX = boundRect[0]
        rectY = boundRect[1]
        rectWidth = boundRect[2]
        rectHeight = boundRect[3]

        # Store and set bounding rect:
        boundingRectangles.append(boundRect)
        color = (0, 0, 255)
        cv2.rectangle(inputImageCopy, (int(rectX), int(rectY)),
                      (int(rectX + rectWidth), int(rectY + rectHeight)), color, 2)

        # Show the final object detection:
        showImage("Objects", inputImageCopy)

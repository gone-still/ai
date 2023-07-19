# File        :   facePreprocessor.py
# Version     :   0.9.5
# Description :   Detects and crops faces from images. To be used for
#                 faceNet training and testing.

# Date:       :   Jul 18, 2023
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


# Check the variance of an image:
def varianceDetector(inputImage, blurThreshold=100):
    # BGR to gray:
    grayImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
    laplacianImage = cv2.Laplacian(grayImage, ddepth=cv2.CV_64F)
    (mean, stdDev) = cv2.meanStdDev(laplacianImage)

    variance = stdDev[0] * stdDev[0]

    print("Variance", variance)

    if variance <= blurThreshold:
        return True
    else:
        return False


# Checks eyes:
def detectEyes(inputImage, imageSize, eyesDetector, displayImage):
    # Resize image:
    eyesImage = cv2.resize(inputImage, imageSize)
    grayImage = cv2.cvtColor(eyesImage, cv2.COLOR_BGR2GRAY)

    # Run face detector:
    eyeROIs = eyesDetector.detectMultiScale(grayImage, scaleFactor=1.05, minNeighbors=30,
                                            minSize=(30, 30),
                                            # maxSize=(150, 150),
                                            flags=cv2.CASCADE_SCALE_IMAGE)

    totalEyes = len(eyeROIs)
    print("Eyes Found: " + str(totalEyes))

    if displayImage:

        for currentROI in eyeROIs:
            # Get the face ROI:
            faceX = int(currentROI[0])
            faceY = int(currentROI[1])
            faceWidth = int(currentROI[2])
            faceHeight = int(currentROI[3])

            # Draw the face ROI:
            color = faceDetectorColor[detectorIndex]
            cv2.rectangle(eyesImage, (faceX, faceY), ((faceX + faceWidth), (faceY + faceHeight)), color, 2)
            # showImage("Eyes", inputImage)

    return totalEyes, eyesImage


# Set project paths:
projectPath = "D://dataSets//faces//"
datasetPath = projectPath + "celebrities"
outputPath = projectPath + "out"
croppedPath = outputPath + "//cropped//Test"

# Cascade files:
cascadeFiles = [("Default", "haarcascade_frontalface_default.xml"), ("Alt", "haarcascade_frontalface_alt.xml"),
                ("Alt 2", "haarcascade_frontalface_alt2.xml"), ("Profile", "haarcascade_profileface.xml")]

# Cascade colors:
faceDetectorColor = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255)]

# For testing, process just one class:
processAll = False
targetClass = "Uniques"

# Set the random seed:
randomSeed = 420

# Script Options:
sampleSize = (32, 32)
eyesSize = (300, 300)

# Detected face filters:
testVariance = False
bypassEyesTest = False
manualFilter = False

writeCropped = True

# Display every processed image?
displayImages = True

# Load each image path of the dataset:
print("[FaceNet Pre-processor] Loading images...")

# Set random seed:
random.seed(randomSeed)

if processAll:

    # Get list of full subdirectories paths:
    classesDirectories = glob(datasetPath + "//*//", recursive=True)
    classesDirectories.sort()

    # Trim root directory, leave only subdirectories names (these will be the classes):
    rootLength = len(datasetPath)
    classesImages = [dirName[rootLength + 1:-1] for dirName in classesDirectories]

    # Get total classes:
    totalClasses = len(classesImages)

else:

    # Process this specific class:
    classesDirectories = [datasetPath + "//" + targetClass + "//"]
    classesImages = [targetClass]

# Load the cascade file(s):
faceDetectors = []
for cascadeFile in cascadeFiles:
    # Load cascade:
    cascadePath = projectPath + "cascades//" + cascadeFile[1]
    faceDetectors.append(cv2.CascadeClassifier(cascadePath))

totalFaceDetectors = len(faceDetectors)
print("[FaceNet Pre-processor] Face detectors loaded: " + str(totalFaceDetectors))

# Load the eye detector cascade:
cascadePath = projectPath + "cascades//" + "haarcascade_eye_3.xml"
eyeDetector = cv2.CascadeClassifier(cascadePath)

# Load images per class:
for c, currentDirectory in enumerate(classesDirectories):
    # Images for this class:
    imagePaths = list(paths.list_images(currentDirectory))
    totalImages = len(imagePaths)

    # Randomize images?
    # Do not randomize for unique pairs:
    if targetClass != "Uniques":
        random.shuffle(imagePaths)

    currentClass = classesImages[c]
    print("[FaceNet Pre-processor] Class: " + currentClass + " Samples: " + str(totalImages))

    # Set cropped output path:
    writePath = croppedPath + "//" + currentClass + "//"

    # Create output directory:
    directoryExists = os.path.isdir(writePath)
    if not directoryExists:

        print("[FaceNet Pre-processor] Creating Directory: " + writePath)
        os.mkdir(writePath)
        directoryExists = os.path.isdir(writePath)

        if directoryExists:
            print("[FaceNet Pre-processor] Successfully created directory: " + writePath)

    # Pre-process each image:
    sampleCount = 0
    faceCount = 0
    writtenImages = 0

    # Unique pair counter:
    pairCount = 0.5

    # Last saved image name:
    lastSaved = "-"
    lastChar = "-"

    for currentPath in imagePaths:

        # Load the image:
        currentImage = cv2.imread(currentPath)
        currentImageCopy = currentImage.copy()

        # BGR to gray:
        grayImage = cv2.cvtColor(currentImage, cv2.COLOR_BGR2GRAY)

        sampleCount += 1

        # Show image:
        if displayImages:
            showImage("Class: " + currentClass + " Sample: " + str(sampleCount), grayImage)

        # Detect face with each detector:
        # Set default face detector via index:
        detectorIndex = 0
        searchDetector = True
        facesROIs = []
        totalFaces = 0

        while searchDetector:

            # Set face detector:
            faceDetector = faceDetectors[detectorIndex]
            # Attempt to detect a face in the image:
            detectorName = cascadeFiles[detectorIndex][0]

            # Run face detector:
            facesROIs = faceDetector.detectMultiScale(grayImage, scaleFactor=1.05, minNeighbors=32,
                                                      minSize=(50, 50),
                                                      # maxSize=(150, 150),
                                                      flags=cv2.CASCADE_SCALE_IMAGE)

            # Get the total ROIS detected:
            totalFaces = len(facesROIs)
            print("[FaceNet Pre-processor] Detector: " + detectorName + " Total Faces found: " + str(totalFaces))

            # Switch face detector if no detections where found:
            if totalFaces > 0 or detectorIndex == totalFaceDetectors - 1:
                searchDetector = False
            else:
                detectorIndex += 1

        # Break loop if no detections
        if totalFaces == 0:
            continue

        # Set up some variable counters:
        faceCount += 1  # Counts detections
        subfaceCount = 0  # Counts ROIs per detections

        for currentROI in facesROIs:

            subfaceCount += 1

            # Get the face ROI:
            faceX = int(currentROI[0])
            faceY = int(currentROI[1])
            faceWidth = int(currentROI[2])
            faceHeight = int(currentROI[3])

            # Draw the face ROI:
            color = faceDetectorColor[detectorIndex]
            cv2.rectangle(currentImageCopy, (faceX, faceY), ((faceX + faceWidth), (faceY + faceHeight)), color, 2)

            # Crop the face:
            croppedFace = currentImage[faceY:faceY + faceHeight, faceX:faceX + faceWidth]

            # Show the images:
            if displayImages:
                showImage("Face(s) Found", currentImageCopy)
                showImage("Cropped Face", croppedFace)

            # Check eyes:
            totalEyes, eyesImage = detectEyes(croppedFace, eyesSize, eyeDetector, displayImages)

            # Show detected eyes on input:
            imageOk = True
            if displayImages:
                windowName = "Detected Eyes"
                cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
                cv2.imshow(windowName, eyesImage)
                key = cv2.waitKey(0)

                # Inspect and manually discard samples:
                if manualFilter:
                    print(key)
                    if key == ord("e"):  # Press "e" to save image
                        print("[FaceNet Pre-processor] Saving sample to disk...")
                    else:
                        print("[FaceNet Pre-processor] Skipping sample...")
                        imageOk = False

            if testVariance:
                # Resize image:
                varianceInput = cv2.resize(croppedFace, sampleSize)
                showImage("Variance Input", varianceInput)

                # Check variance:
                goodVariance = varianceDetector(croppedFace, 100)

            # Check image type (A/B) for uniques pairs:
            if targetClass == "Uniques":
                if sampleCount % 2 == 1:
                    currentChar = "A"
                else:
                    currentChar = "B"

            # Save image to outputDirectory:
            # Do not save the image if no eyes have been detected
            # Option is overriden by the "eyes check" flag:
            if totalEyes > 0 or bypassEyesTest:
                # Check manual filter:
                if writeCropped and imageOk:

                    # Check class type:
                    if targetClass != "Uniques":

                        # Not unique pair, straightforward file name:
                        imageName = str(faceCount) + "-" + str(subfaceCount)

                    else:

                        # Check that this is truly a pair in the A-B sequence:
                        print("Pair:", currentChar, lastChar)
                        print("Last Saved:", lastSaved)
                        if currentChar == lastChar:
                            # Missed one image from this pair...
                            print("[FaceNet Pre-processor] Skipped a pair image...")

                            print(pairCount)

                            # Delete last saved...
                            if currentChar == "A":
                                print("[FaceNet Pre-processor] Deleting last saved...: " + lastSaved)
                                os.remove(lastSaved)
                                pairCount = pairCount - 0.5
                                # continue

                            else:
                                # Buffer the char used:
                                lastChar = "-"
                                print("[FaceNet Pre-processor] Skipping image...")
                                # Abort:
                                continue

                        # Set name for unique pair:
                        imageName = str(math.ceil(pairCount)) + "-" + currentChar
                        pairCount = pairCount + 0.5
                        # Buffer the char used:
                        lastChar = currentChar

                    # Write the image:
                    imagePath = writePath + "pair-" + imageName + ".png"
                    # Save the last file path written:
                    lastSaved = imagePath
                    writeImage(imagePath, croppedFace)
                    writtenImages += 1

    # Get a couple of stats:
    detectionRate = (faceCount / totalImages) * 100
    writtenRate = (writtenImages / totalImages) * 100

    print("[FaceNet Pre-processor] Detected faces rate: " + str(faceCount) + "/" + str(
        totalImages) + " [" + f"{detectionRate:.4f}" + "%]" + " Written: " + str(writtenImages) + "/" + str(
        totalImages) + " [" + f"{writtenRate:.4f}" + "%]")

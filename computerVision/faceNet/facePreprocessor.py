# File        :   facePreprocessor.py
# Version     :   0.9.9
# Description :   Detects and crops faces from images. To be used for
#                 faceNet training and testing.

# Date:       :   Jul 27, 2023
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT

import random

import cv2
import os
import math

import mtcnn

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
def detectEyes(mtcnnEyes, inputImage, imageSize, eyesDetector, displayImage=False, eyesColor=(0, 255, 0)):
    # Resize image:
    originalHeight, originalWidth = inputImage.shape[0:2]
    eyesImage = cv2.resize(inputImage, imageSize)

    # Default scale (no scale):
    xScale = 1.0
    yScale = 1.0

    if not mtcnnEyes:
        # Detect eyes via cascade:
        grayImage = cv2.cvtColor(eyesImage, cv2.COLOR_BGR2GRAY)

        # Run face detector:
        eyeROIs = eyesDetector.detectMultiScale(grayImage, scaleFactor=1.05, minNeighbors=30,
                                                minSize=(30, 30),
                                                # maxSize=(150, 150),
                                                flags=cv2.CASCADE_SCALE_IMAGE)
    else:
        # Eyes already detected by mtcnnEyes (if any):
        eyeROIs = mtcnnEyes
        # Resize them ROIs:
        xScale = imageSize[0] / originalWidth
        yScale = imageSize[1] / originalHeight

    totalEyes = len(eyeROIs)
    print("Eyes Found: " + str(totalEyes))

    if displayImage:

        for currentROI in eyeROIs:
            # Get the face ROI:
            faceX = int(currentROI[0] * xScale)
            faceY = int(currentROI[1] * yScale)
            faceWidth = int(currentROI[2] * xScale)
            faceHeight = int(currentROI[3] * yScale)

            # Draw the face ROI:
            # color = faceDetectorColor[detectorIndex]
            cv2.rectangle(eyesImage, (faceX, faceY), ((faceX + faceWidth), (faceY + faceHeight)), eyesColor, 2)
            # showImage("Eyes", eyesImage)

    return totalEyes, eyesImage


# Rotates an image by a given angle (degs):
def rotateImage(inputImage, angle, imageCenter=None):
    # Grab the dimensions of the image and calculate the center of the
    # image
    (h, w) = inputImage.shape[:2]
    if not imageCenter:
        (cX, cY) = (w // 2, h // 2)
    else:
        cX, cY = (imageCenter[0], imageCenter[1])
    # rotate our image by 45 degrees around the center of the image
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    rotatedImage = cv2.warpAffine(inputImage, M, (w, h))
    return rotatedImage


# Set project paths:
projectPath = "D://dataSets//faces//"
datasetPath = projectPath + "celebrities"
outputPath = projectPath + "out"
croppedPath = outputPath + "//cropped//"

# Cascade files:
cascadeFiles = [("Default", "haarcascade_frontalface_default.xml"), ("Alt", "haarcascade_frontalface_alt.xml"),
                ("Alt 2", "haarcascade_frontalface_alt2.xml"), ("Profile", "haarcascade_profileface.xml")]

# Cascade colors:
faceDetectorColor = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255), (192, 0, 192)]

# For testing, process just one class:
processAll = False
targetClass = "Uniques"

# Set the random seed:
randomSeed = 420

# Script Options:
sampleSize = (32, 32)
eyesSize = (300, 300)

# Eye proportions (from eye centroid to eye bbox via mtcnn):
wRatio = 0.28
hRatio = 0.13

# Detected face filters:
testVariance = False
bypassEyesTest = False
manualFilter = False
rotateCrop = True

writeCropped = True

# Display every processed image?
displayImages = False

# Save png version of input?
# (ignore if the image is already png)
savePng = True
deleteOriginal = True

# Load each image path of the dataset:
print("[FaceNet Pre-processor] Loading images...")

# mtcnn detector:
detector = mtcnn.MTCNN()

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

# Get total of detectors used in extractor:
maxDetectors = totalFaceDetectors - 1

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
    writePath = croppedPath + currentClass + "//"

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

        # Save png version:
        if savePng:
            # Get file name:
            filenameString = currentPath.split(".")
            # Check extension before saving & deleting:
            if filenameString[1] != "png":
                filenameString = filenameString[0] + ".png"
                print("[FaceNet Pre-processor] Writing PNG image: "+filenameString)
                writeImage(filenameString, currentImage)
                # Remove original image:
                if deleteOriginal:
                    print("[FaceNet Pre-processor] Removing File: "+currentPath)
                    os.remove(currentPath)

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

        # mtcnn flag:
        useMtcnn = False
        # mtcnn eyes info:
        mtcnnEyes = []

        while searchDetector:

            if not useMtcnn:

                # Set face detector:
                faceDetector = faceDetectors[detectorIndex]
                # Attempt to detect a face in the image:
                detectorName = cascadeFiles[detectorIndex][0]

                # Run face detector:
                facesROIs = faceDetector.detectMultiScale(grayImage, scaleFactor=1.05, minNeighbors=32,
                                                          minSize=(50, 50),
                                                          # maxSize=(150, 150),
                                                          flags=cv2.CASCADE_SCALE_IMAGE)

            else:

                # Set the deep face detector:
                detectorName = "MTCNN"

                # Store original width and height:
                originalHeight, originalWidth = currentImage.shape[0:2]

                # BGR to RGB:
                rgbImage = cv2.cvtColor(currentImage, cv2.COLOR_BGR2RGB)

                # Run detector:
                faces = detector.detect_faces(rgbImage)

                # faces is an array with all the bounding boxes detected.
                # Store results in a list:
                for face in faces:

                    # Get coordinates from dict:
                    x, y, w, h = face["box"]

                    # Into the list:
                    facesROIs = list(facesROIs)
                    facesROIs.append([x, y, w, h])

                    # Get eyes info:
                    keypoints = face["keypoints"]

                    # Compute bb width and height:
                    eyeWidth = wRatio * w
                    eyeHeight = hRatio * h

                    # Process both eyes:
                    for currentKeypoint in ["left_eye", "right_eye"]:
                        # Get eyes:
                        currentEyes = keypoints[currentKeypoint]

                        # Get centroids (local on cropped image):
                        eyeX = currentEyes[0] - x
                        eyeY = currentEyes[1] - y

                        # Compute top left corner:
                        leftX = eyeX - 0.5 * eyeWidth
                        leftY = eyeY - 0.5 * eyeHeight

                        # Into the list:
                        mtcnnEyes.append([int(leftX), int(leftY), int(eyeWidth), int(eyeHeight)])

                # Set detector index (mtcnn is last one)
                detectorIndex += 1
                searchDetector = False

            # Get the total ROIS detected:
            totalFaces = len(facesROIs)
            print("[FaceNet Pre-processor] Detector: " + detectorName + " Total Faces found: " + str(totalFaces))

            # Switch face detector if no detections where found:
            if totalFaces > 0:
                searchDetector = False
            else:
                if detectorIndex < maxDetectors:
                    detectorIndex += 1
                else:
                    print("[FaceNet Pre-processor] Switching to mtcnn face detector...")
                    useMtcnn = True

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
            totalEyes, eyesImage = detectEyes(mtcnnEyes, croppedFace, eyesSize, eyeDetector, displayImages, color)

            # Check image variance:
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

            # Rotate output image?:
            if rotateCrop:
                randomAngle = random.randint(-10, 10)
                print("[FaceNet Pre-processor] Rotating cropped face by: " + str(randomAngle) + " degs.")
                bboxCx = faceX + 0.5 * faceWidth
                bboxCy = faceY + 0.5 * faceHeight
                imageCenter = (bboxCx, bboxCy)
                currentImage = rotateImage(currentImage, randomAngle, imageCenter)
                croppedFace = currentImage[faceY:faceY + faceHeight, faceX:faceX + faceWidth]
                if displayImages:
                    showImage("Rotated", croppedFace)

            # Show detected eyes on input:
            imageOk = True
            if displayImages or totalEyes == 0:
                continueCondition = (manualFilter or (not bypassEyesTest and totalEyes == 0))
                windowName = "Detected Eyes"

                cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
                cv2.imshow(windowName, eyesImage)

                if continueCondition:
                    print("[FaceNet Pre-processor] Press \"e\" to save image...")

                key = cv2.waitKey(0)

                # Inspect and manually discard samples:
                if continueCondition:
                    print("[FaceNet Pre-processor] Pressed: " + str(key))
                    if key == ord("e"):  # Press "e" to save image
                        print("[FaceNet Pre-processor] Saving sample to disk...")
                        totalEyes = 2
                    else:
                        print("[FaceNet Pre-processor] Skipping sample...")
                        imageOk = False

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

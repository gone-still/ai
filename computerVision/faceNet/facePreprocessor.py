# File        :   facePreprocessor.py
# Version     :   0.15.3
# Description :   Detects and crops faces from images. To be used for
#                 faceNet training and testing.

# Date:       :   Aug 18, 2023
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT

import random

import cv2
import os
import math
import numpy as np

from natsort import os_sorted
from pathlib import Path

import time

import mtcnn


# Read an image:
def readImage(imagePath):
    # Loads image:
    inputImage = cv2.imread(imagePath)
    # Checks if image was successfully loaded:
    if inputImage is None:
        raise TypeError("readImage>> Error: Could not load Input image.")

    return inputImage


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
    print(eyeROIs)
    print("Unfiltered Eyes Found: " + str(totalEyes))

    # Post process eyes,
    # Approximate each eye bounding box area:
    eyeArea = []
    maxArea = 0
    for currentROI in eyeROIs:

        # Get the eye ROI:
        # eyeX = int(currentROI[0] * xScale)
        # eyeY = int(currentROI[1] * yScale)
        eyeWidth = int(currentROI[2] * xScale)
        eyeHeight = int(currentROI[3] * yScale)

        # Compute area:
        currentEyeArea = eyeWidth * eyeHeight
        eyeArea.append(currentEyeArea)

        # Get max area:
        if currentEyeArea > maxArea:
            maxArea = currentEyeArea

    # Set minimum area:
    minEyeArea = 0.8 * maxArea
    totalEyes = 0
    # Filter eyes:
    for i in range(len(eyeROIs)):

        # Get area:
        currentEyeArea = eyeArea[i]

        # Filter:
        eyeColor = (0, 0, 255)
        if currentEyeArea > minEyeArea:
            eyeColor = (0, 255, 0)
            totalEyes += 1

        if displayImage:
            # Get current ROI:
            eyeROI = eyeROIs[i]
            # Get the face ROI:
            eyeX = int(eyeROI[0] * xScale)
            eyeY = int(eyeROI[1] * yScale)
            eyeWidth = int(eyeROI[2] * xScale)
            eyeHeight = int(eyeROI[3] * yScale)
            # Draw the face ROI:
            cv2.rectangle(eyesImage, (eyeX, eyeY), ((eyeX + eyeWidth), (eyeY + eyeHeight)), eyeColor, 2)
            showImage("Eyes", eyesImage)

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
croppedPath = outputPath + "//cropped//Train//"

# Cascade files:
cascadeNames = ["Default 1", "Default 2", "Alt", "Alt 2", "Profile", "MTCNN"]

# Cascade files with rectangle color:
cascadeDict = {
    cascadeNames[0]: {"File": "haarcascade_frontalface_default.xml", "Color": (0, 255, 0)},
    cascadeNames[1]: {"File": "haarcascade_frontalface_default.xml", "Color": (0, 130, 0)},
    cascadeNames[2]: {"File": "haarcascade_frontalface_alt.xml", "Color": (255, 0, 0)},
    cascadeNames[3]: {"File": "haarcascade_frontalface_alt2.xml", "Color": (0, 0, 255)},
    cascadeNames[4]: {"File": "haarcascade_profileface.xml", "Color": (0, 255, 255)}
}

# Cascade parameters:
cascadeParams = {
    cascadeNames[0]: {"scaleFactor": 1.05, "minNeigh": 50, "minSize": 70, "maxSize": -1},
    cascadeNames[1]: {"scaleFactor": 1.01, "minNeigh": 150, "minSize": 90, "maxSize": -1},
    cascadeNames[2]: {"scaleFactor": 1.01, "minNeigh": 80, "minSize": 70, "maxSize": -1},
    cascadeNames[3]: {"scaleFactor": 1.01, "minNeigh": 90, "minSize": 70, "maxSize": -1},
    cascadeNames[4]: {"scaleFactor": 1.01, "minNeigh": 10, "minSize": 300, "maxSize": 500}
}

# For testing, process just one class:
targetClasses = ["Uniques"]

# Uniques dir suffix:
dirSuffix = ""  # " Test"

# Set the random seed:
randomSeed = 420

# Script Options:
sampleSize = (32, 32)
eyesSize = (300, 300)

# Eye proportions (from eye centroid to eye bbox via mtcnn):
wRatio = 0.28
hRatio = 0.13

# Run face detectors?
detectFaces = True

# Fresh start/add new samples to existing directory:
resumeCropping = True

# Detected face filters:
testVariance = False
bypassEyesTest = False
manualFilter = False
rotateCrop = True

# Show cropped face when no eyes are detected?
displayFilter = True

# Write cropped face to output dir?
writeCropped = True

# Display every processed image?
displayImages = False
# Display just final (to-be-written) image?
displayFinal = False

# Process images alphabetically by file name?
# True -> alphabetically, False -> creation date
alphabeticalSort = False

# Save png version of input?
savePng = True

# Ignore if file is already png? (forced png writing)
overwritePng = False
deleteOriginal = True

# Rename original files to "sample/pair":
renameFiles = True

# File name suffix (sample or pair):
fileNameSufix = "-"

# Stats dictionary:
detectorsStats = {}

# Set the stats dictionary
for i in range(len(cascadeNames)):
    currentName = cascadeNames[i]
    detectorsStats[currentName] = [0.0, 0.0]

# Missed detections dictionary:
missedDetections = {}

# Load each image path of the dataset:
print("[FaceNet Pre-processor] Loading images...")

# mtcnn detector:
detector = mtcnn.MTCNN()

# Set random seeds:
random.seed(randomSeed)
np.random.seed(randomSeed)

# Process the list of classes:
classesDirectories = [datasetPath + "//" + targetClass + dirSuffix + "//" for targetClass in targetClasses]
classesImages = targetClasses

# Get total classes:
totalClasses = len(classesImages)

# Load the cascade file(s):
faceDetectors = []
for cascadeName in cascadeDict:
    # Get the file
    cascadeFile = cascadeDict[cascadeName]["File"]
    # Load cascade:
    cascadePath = projectPath + "cascades//" + cascadeFile
    faceDetectors.append([cv2.CascadeClassifier(cascadePath), cascadeName])

totalFaceDetectors = len(faceDetectors)
print("[FaceNet Pre-processor] Face detectors loaded: " + str(totalFaceDetectors))

# Get total of detectors used in extractor:
maxDetectors = totalFaceDetectors - 1

# Load the eye detector cascade:
cascadePath = projectPath + "cascades//" + "haarcascade_eye_3.xml"
eyeDetector = cv2.CascadeClassifier(cascadePath)

# Counter for all the images written to disk:
totalImagesWritten = 0

# Global counters (totalDetections, writtenImages)
processCounters = {"totalDetections": 0.0, "writtenImages": 0.0}

# Load images per class:
for c, currentDirectory in enumerate(classesDirectories):

    # Get class:
    currentClass = classesImages[c]
    print("[FaceNet Pre-processor] Processing Class: " + currentClass +
          " [" + str(c + 1) + "/" + str(totalClasses) + "]")
    print("[FaceNet Pre-processor] Class path: " + currentDirectory)

    # Set cropped output path:
    writePath = croppedPath + currentClass + dirSuffix + "//"
    print("[FaceNet Pre-processor] Target path: " + writePath)

    # Create output directory:
    directoryExists = os.path.isdir(writePath)
    if not directoryExists:

        print("[FaceNet Pre-processor] Creating Directory: " + writePath)
        os.mkdir(writePath)
        directoryExists = os.path.isdir(writePath)

        if directoryExists:
            print("[FaceNet Pre-processor] Successfully created directory: " + writePath)

    # Check out target directory, if resume cropping is enabled, keep writing sample where
    # the last sample left off...
    latestFileTime = -1
    lastSavedFileNumber = 0
    if directoryExists:
        # Get list of already created croppings:
        existingImages = sorted(Path(writePath).iterdir(), key=os.path.getctime)

        # Check if empty list:
        if not existingImages:
            print("Found no existent images in target directory. Saving from scratch...")
            # Can't resume cropping without previously cropped images:
            resumeCropping = False
        else:
            # Get last file:
            lastFile = existingImages[-1]

            # Get number of last saved file:
            lastSavedFileNumber = int(lastFile.stem.split("-")[1])
            print("Starting file number: " + str(lastSavedFileNumber))

            # Get creation time of last file:
            latestFileTime = os.path.getctime(lastFile)
            print("[FaceNet Pre-processor] Last created sample: " + lastFile.stem + " Time: " + time.ctime(
                latestFileTime) + " (" + str(latestFileTime) + ")")

    # Images for this class:
    if alphabeticalSort:
        # Get and store files alphabetically:
        print("[FaceNet Pre-processor] Sorting images alphabetically...")
        imagePaths = os_sorted(Path(currentDirectory).iterdir())

    else:
        # Get and store files according to creation date:
        print("[FaceNet Pre-processor] Sorting images by creation date...")
        imagePaths = sorted(Path(currentDirectory).iterdir(), key=os.path.getctime)

    totalImages = len(imagePaths)
    print("Total Class files: " + str(totalImages))

    # Prepare the random vector of names for non-unique classes:
    if currentClass != "Uniques":
        # Randomly create images list for output:
        choicesArray = np.arange(1, totalImages + 1, 1, dtype=int)
        choicesArray = np.random.choice(choicesArray, len(choicesArray), replace=False)
        choicesDict = {}

    # Create filtered dict:
    if resumeCropping:
        filteredDict = {}

    # Create filenames dictionary:
    filenamesDict = {}
    # Skipped images counter:
    skippedImages = 0
    for i, f in enumerate(imagePaths):

        # Get filename only (without extension):
        currentFilename = f.stem

        # Get file time:
        fileTime = os.path.getctime(f)

        # Into the dictionary:
        if currentClass != "Uniques":
            # Sample out name will only be the image count:
            fileNameSufix = "sample"
            # Choose random generated out number from choices array:
            randomChoice = choicesArray[i]
            # imageCount = i + 1
            outName = fileNameSufix + "-" + str(randomChoice)
            # Check non-repetitive:
            if outName not in choicesDict:
                choicesDict[outName] = 1
            else:
                print("Found repeated random value...")
            filenamesDict[currentFilename] = outName
        else:
            # Set pair number:
            pairCount = math.ceil(0.5 * (i + 1))
            # Set pair char:
            if (i + 1) % 2 == 1:
                pairChar = "A"
            else:
                pairChar = "B"

            # Set outfile name according to pair "rule":
            fileNameSufix = "pair"
            filenamesDict[currentFilename] = fileNameSufix + "-" + str(pairCount) + "-" + pairChar

        # if resume enabled, discard all images that where created BEFORE the latest file:
        if resumeCropping:
            print(i, currentFilename, fileTime, latestFileTime)
            if fileTime > latestFileTime:
                filteredDict[currentFilename] = filenamesDict[currentFilename]
                print(str(i) + " New File: " + currentFilename)
            else:
                skippedImages += 1

    # If cropping enabled...
    if resumeCropping:
        print("Total skipped images: " + str(skippedImages))
        print("Total new images: " + str(len(filteredDict)))
        print("Total: " + str(totalImages))
        # Set the filenames dict to the filtered dict:
        filenamesDict = filteredDict
        # Set new images paths list:
        imagePaths = imagePaths[skippedImages:]
        # Set new total images count:
        totalImages = len(imagePaths)
        print("First image: " + str(imagePaths[0]))

    currentClass = classesImages[c]
    print("[FaceNet Pre-processor] Class: " + currentClass + " Samples: " + str(totalImages))

    # Prepare missed detections log for this class:
    missedDetections[currentClass] = []

    # Pre-process each image:
    sampleCount = 0
    faceCount = 0
    writtenImages = 0
    totalDetections = 0

    # Last saved image name:
    lastSaved = "-"
    lastChar = "*"

    # Processed files counter:
    filesProcessed = 0

    for j, currentPath in enumerate(imagePaths):
        # Processed files counter goes up:
        filesProcessed += 1

        # Get absolute path:
        currentPath = os.path.abspath(currentPath)

        # Get filename (without extension):
        currentFilename = imagePaths[j].stem
        # Get extension:
        currentExtension = imagePaths[j].suffix

        # Print info:
        print(">> [" + str(j + 1) + "/" + str(totalImages) + "] Image: " + str(currentPath),
              "Date: " + str(time.ctime(os.path.getctime(currentPath))))

        # Load the image:
        currentImage = readImage(currentPath)
        currentImageCopy = currentImage.copy()

        # Check if original file must be overwritten (should be either sample or pair):
        currentFilenamePrefix = currentFilename.split("-")[0]

        # Rewrite original files is flag is set or image name is different from processed
        # suffix: either "sample" or "pair"
        if overwritePng or currentFilenamePrefix != fileNameSufix:

            # Remove original image:
            if deleteOriginal:
                print("[FaceNet Pre-processor] Removing File: " + currentPath)
                os.remove(currentPath)

            # Create out name:
            parentDir = str(imagePaths[j].parent) + "//"
            if renameFiles:
                currentFilename = filenamesDict[currentFilename]

            filenameString = parentDir + currentFilename + ".png"

            # Save png version:
            if savePng:
                print("[FaceNet Pre-processor] Writing PNG image: " + filenameString)
                if os.path.exists(filenameString):
                    print("File: " + filenameString + "Already exists. Renaming new file...")
                    filenameString = parentDir + currentFilename + "_new" + ".png"
                writeImage(filenameString, currentImage)

        # BGR to gray:
        grayImage = cv2.cvtColor(currentImage, cv2.COLOR_BGR2GRAY)
        sampleCount += 1

        # Show image:
        if displayImages:
            showImage("Class: " + currentClass + " Sample: " + str(sampleCount), grayImage)

        # Detect faces?
        if not detectFaces:
            continue

        # Detect face with each detector:
        # Set default face detector via index:
        detectorIndex = 0
        searchDetector = True
        facesROIs = []
        totalFaces = 0
        detectorName = ""

        # mtcnn flag:
        useMtcnn = False
        # mtcnn eyes info:
        mtcnnEyes = []

        while searchDetector:

            if not useMtcnn:

                # Set face detector & name:
                faceDetector = faceDetectors[detectorIndex][0]
                detectorName = faceDetectors[detectorIndex][1]

                # Attempt to detect a face in the image:
                print("[FaceNet Pre-processor] Using detector: " + detectorName)

                # Get params
                currentCascadeParams = cascadeParams[detectorName]

                # Set params:
                scaleFactor = currentCascadeParams["scaleFactor"]
                minNeighbors = currentCascadeParams["minNeigh"]
                minSize = (currentCascadeParams["minSize"], currentCascadeParams["minSize"])

                # Set max Size:
                maxSize = currentCascadeParams["maxSize"]
                if maxSize == -1:
                    maxSize = None
                else:
                    maxSize = (maxSize, maxSize)

                # Run face detector:
                facesROIs = faceDetector.detectMultiScale(grayImage, scaleFactor=scaleFactor, minNeighbors=minNeighbors,
                                                          minSize=minSize,
                                                          maxSize=maxSize,
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
                    print("[FaceNet Pre-processor] Switching to MTCNN face detector...")
                    useMtcnn = True

        # Break loop if no detections
        if totalFaces == 0:
            # Log missed detections,
            # Do not duplicate entries:
            if currentFilename not in missedDetections[currentClass]:
                missedDetections[currentClass].append(currentFilename)
            # Abort iteration:
            continue

        # Set up some variable counters:
        faceCount += 1  # Counts detections
        subfaceCount = 0  # Counts ROIs per detections

        # Found valid subface flag:
        foundValidFace = False

        # Check every found ROI:
        for currentROI in facesROIs:
            # Increase face counter in this image:
            subfaceCount += 1

            # Increase detections counter:
            totalDetections += 1

            # Store some stats. Face detection counter for this particular detector:
            detectorsStats[detectorName][0] += 1

            # Get the face ROI:
            faceX = int(currentROI[0])
            faceY = int(currentROI[1])
            faceWidth = int(currentROI[2])
            faceHeight = int(currentROI[3])

            # Draw the face ROI:
            if detectorName != "MTCNN":
                color = cascadeDict[detectorName]["Color"]
            else:
                color = (192, 0, 192)
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
            if currentClass == "Uniques":
                if sampleCount % 2 == 1:
                    currentChar = "A"
                else:
                    currentChar = "B"

            # Rotate output image?:
            if rotateCrop:
                randomAngle = random.randint(-10, 10)
                print("[FaceNet Pre-processor] Rotating cropped face by: " + str(randomAngle) + " degs.")
                # Compute Centroid:
                bboxCx = faceX + 0.5 * faceWidth
                bboxCy = faceY + 0.5 * faceHeight
                imageCenter = (bboxCx, bboxCy)
                # Rotate the image:
                rotatedImage = rotateImage(currentImage, randomAngle, imageCenter)
                # Crop face:
                croppedFace = rotatedImage[faceY:faceY + faceHeight, faceX:faceX + faceWidth]
                # Show rotated face:
                if displayImages:
                    showImage("Rotated", croppedFace)

            # Show detected eyes on input:
            imageOk = True
            if displayFilter and (displayImages or totalEyes == 0):
                continueCondition = (manualFilter or (not bypassEyesTest and totalEyes == 0))
                windowName = "Eyes Filter"

                cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
                cv2.imshow(windowName, eyesImage)

                if continueCondition:
                    print("[FaceNet Pre-processor] Press \"e\" to save image...")

                key = cv2.waitKey(0)

                # Inspect and manually discard samples:
                if continueCondition:
                    print("[FaceNet Pre-processor] Pressed: " + str(key))
                    if key == (ord("e") or ord("E")):  # Press "e" to save image
                        print("[FaceNet Pre-processor] Saving sample to disk...")
                        totalEyes = 2
                    else:
                        print("[FaceNet Pre-processor] Skipping sample...")
                        imageOk = False

            # Save image to outputDirectory:
            # Do not save the image if no eyes have been detected
            # Option is overriden by the "eyes check" flag:
            if detectorName != "MTCNN":
                maxEyes = 4
            else:
                maxEyes = 2

            eyesCondition = maxEyes >= totalEyes > 0

            if eyesCondition or bypassEyesTest:

                # Check manual filter:
                if imageOk:

                    # Check class type:
                    if currentClass != "Uniques":

                        # Not unique pair, straightforward file name:
                        # imageName = str(faceCount) + "-" + str(subfaceCount)
                        imageName = filenamesDict[imagePaths[j].stem] + "-" + str(subfaceCount)

                    else:

                        # Check that this is truly a pair in the A-B sequence:
                        print("[FaceNet Pre-processor] Pair:", currentChar, lastChar)
                        print("[FaceNet Pre-processor] Last Saved:", lastSaved)
                        if (currentChar == lastChar) or (currentChar == "B" and lastChar == "*"):

                            # Missed one image from this pair...
                            print("[FaceNet Pre-processor] Skipped a pair image...")

                            # Delete last saved...
                            if currentChar == "A":
                                print("[FaceNet Pre-processor] Deleting last saved...: " + lastSaved)
                                os.remove(lastSaved)

                            else:
                                # Buffer the char used:
                                lastChar = "-"
                                print("[FaceNet Pre-processor] Skipping image...")

                                # Abort:
                                continue

                        # Set name for unique pair:
                        imageName = currentFilename

                        # Buffer the char used:
                        lastChar = currentChar

                    # Write the image:
                    imagePath = writePath + imageName + ".png"
                    # Save the last file path written:
                    lastSaved = imagePath

                    # Found valid face:
                    foundValidFace = True

                    print("Writing image: " + imagePath)

                    if writeCropped:
                        writeImage(imagePath, croppedFace)
                        writtenImages += 1

                    # Show final image?
                    if displayFinal:
                        showImage("Out Image", croppedFace)

                    # Store some stats. Actual images written counter for this particular detector:
                    detectorsStats[detectorName][1] += 1

        # Log failed detection if valid subface was not found:
        if not foundValidFace:
            # Do not duplicate entries:
            if currentFilename not in missedDetections[currentClass]:
                missedDetections[currentClass].append(currentFilename)

    # Get a couple of stats:
    detectionRate = (faceCount / totalImages) * 100
    writtenRate = (writtenImages / totalImages) * 100

    # Add to total images written:
    totalImagesWritten += writtenImages

    # Store process counters:
    processCounters["totalDetections"] += totalDetections
    processCounters["writtenImages"] += writtenImages

    print(currentClass + " - Processed files: " + str(filesProcessed))
    print(">> [FaceNet Pre-processor] Detected faces rate: " + str(faceCount) + "/" + str(
        totalImages) + " [" + f"{detectionRate:.4f}" + "%]" + " Written: " + str(writtenImages) + "/" + str(
        totalImages) + " [" + f"{writtenRate:.4f}" + "%]")

# Print stats per detector:
print("\nDetector Stats:")
for currentDetector in detectorsStats:
    # Get the stats
    statsList = detectorsStats[currentDetector]
    facesDetected = statsList[0]
    facesWritten = statsList[1]

    # Compute the rates:
    detectionRate = 0.0
    writtenRate = 0.0
    if processCounters["totalDetections"] > 0:
        detectionRate = (facesDetected / processCounters["totalDetections"]) * 100

    if processCounters["writtenImages"] > 0:
        writtenRate = (facesWritten / processCounters["writtenImages"]) * 100

    # Set the strings:
    detectionPercent = f"{detectionRate:.2f}" + "%"
    writtenPercent = f"{writtenRate:.2f}" + "%"

    # Pretty print the detectors stats:
    print('{0:<20}: {1:<15} {2:<15} {3:<15} {4:<15}'.format("[" + currentDetector + "]",
                                                            "Detected: " + str(facesDetected),
                                                            "Written: " + str(facesWritten),
                                                            "D(%): " + detectionPercent,
                                                            "W(%): " + writtenPercent))

# Print missed detections per image:
print("\nMissed Detections:")
totalMissedDetections = 0
for currentClass in missedDetections:
    print("Class: " + currentClass)
    missedList = missedDetections[currentClass]
    for i, currentImage in enumerate(missedList):
        # Show the info:
        print(" ", i, currentImage)
        # Accumulate missed images:
        totalMissedDetections += 1

print("Total Missed Detections: ", totalMissedDetections)

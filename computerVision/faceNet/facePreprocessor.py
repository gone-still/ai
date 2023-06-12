import cv2
import os
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


# Set project paths:
projectPath = "D://dataSets//faces//"
datasetPath = projectPath + "celebrities"
outputPath = projectPath + "out"
croppedPath = outputPath + "//cropped"

# Cascade files:
cascadeFiles = [("Default", "haarcascade_frontalface_default.xml"), ("Alt", "haarcascade_frontalface_alt.xml"),
                ("Alt 2", "haarcascade_frontalface_alt2.xml"), ("Profile", "haarcascade_profileface.xml")]

# Cascade colors:
faceDetectorColor = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255)]

# For testing, process just one class:
processAll = False
targetClass = "Will Smith"

# Script Options:
displayImages = True
writeCropped = True

# Load each image path of the dataset:
print("[FaceNet Pre-processor] Loading images...")

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

# Load images per class:
for c, currentDirectory in enumerate(classesDirectories):
    # Images for this class:
    imagePaths = list(paths.list_images(currentDirectory))
    totalImages = len(imagePaths)

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
        # for faceDetector in faceDetectors:

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
            print("Detector: " + detectorName + " Total Faces found: " + str(totalFaces))

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

            # Save image to outputDirectory:
            if writeCropped:
                imagePath = writePath + "sample-" + str(faceCount) + "-" + str(subfaceCount) + ".png"
                writeImage(imagePath, croppedFace)

    detectionRate = (faceCount / totalImages) * 100
    print("[FaceNet Pre-processor] Detected faces rate: " + str(faceCount) + "/" + str(
        totalImages) + " [" + f"{detectionRate:.4f}" + "%]")

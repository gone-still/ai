# File        :   faceTest.py
# Version     :   0.9.2
# Description :   faceNet test script

# Date:       :   Jul 03, 2023
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT


# Defines a re-sizable image window:
def showImage(imageName, inputImage):
    cv2.namedWindow(imageName, cv2.WINDOW_NORMAL)
    cv2.imshow(imageName, inputImage)
    cv2.waitKey(0)


# Set project paths:
projectPath = "D://dataSets//faces//"
outputPath = projectPath + "out//"
datasetPath = outputPath + "cropped"
weightsFilename = "facenetWeights.h5"

# Total positive pairs & negative pairs to be tested:
startImage = 51  # Start at this image for test
maxImages = 30  # Use all remaining images for test
datasetSize = 150
positivePortion = 0.7

randomSeed = 420
pairsPerClass = 200

displayImages = False
showClassified = True

# Set the DNN's parameters:
imageDims = (100, 100, 3)
resizeInterpolation = cv2.INTER_AREA

embeddingSize = 512
lr = 0.002
trainingEpochs = 1

# Set the image dimensions:
imageHeight = imageDims[0]
imageWidth = imageDims[1]
imageChannels = imageDims[2]

# Set random seed:
random.seed(randomSeed)
np.random.seed(randomSeed)

# Read the test images:
# Get list of full subdirectories paths:
classesDirectories = glob(datasetPath + "//*//", recursive=True)
classesDirectories.sort()

# Trim root directory, leave only subdirectories names (these will be the classes):
rootLength = len(datasetPath)
classesImages = [dirName[rootLength + 1:-1] for dirName in classesDirectories]

# The classes dictionary:
classesDictionary = {}

# Create classes dictionary:
classCounter = 0
for c in classesImages:
    if c not in classesDictionary:
        classesDictionary[c] = classCounter
        classCounter += 1

print(classesDictionary)

# Get total classes:
totalClasses = len(classesImages)
print("Total Classes:", totalClasses, classesImages)

# Store the samples total here:
totalDatasetSamples = 0
imagesRange = 0

# Create the test dataset as a dictionary:
testDataset = {}

# Load images per class:
for c, currentDirectory in enumerate(classesDirectories):
    # Images for this class:
    imagePaths = list(paths.list_images(currentDirectory))
    totalImages = len(imagePaths)

    # Slice test images:
    if maxImages == -1:
        imagePaths = imagePaths[startImage:]
    else:
        # Compute last image:
        samplesDiff = totalImages - startImage
        if samplesDiff < maxImages:
            endImage = totalImages
        else:
            endImage = startImage + maxImages
        imagePaths = imagePaths[startImage:endImage]

    # Get class:
    currentClass = classesImages[c]
    testSamples = len(imagePaths)

    # Check for minimum number of necessary images to create
    # a unique pair:
    if testSamples < 2:
        print(
            "[FaceNet Test] [X] Skipping class: " + currentClass + " due to insufficient samples to build pairs. Test Samples: " + str(
                testSamples))
        continue

    print("[FaceNet Test] Class: " + currentClass + " Using: " + str(testSamples) + " samples for testing...")

    # Create dictionary key:
    # Each key is a class name associated with
    # an array of samples for this class:
    testDataset[currentClass] = []

    # Load the full images:
    imagesRange = testSamples

    # Load the images:
    for p in range(imagesRange):
        # for currentPath in imagePaths:
        # Get current path:
        currentPath = imagePaths[p]

        # Load the image:
        currentImage = cv2.imread(currentPath)

        # Should it be converted to grayscale (one channel):
        targetDepth = imageDims[-1]

        if targetDepth != 3:
            # To Gray:
            currentImage = cv2.cvtColor(currentImage, cv2.COLOR_BGR2GRAY)

        # Pre-process the image for FaceNet input:
        newSize = imageDims[0:2]

        # Resize:
        currentImage = cv2.resize(currentImage, newSize, resizeInterpolation)

        # Scale:
        currentImage = currentImage.astype("float") / 255.0

        if targetDepth == 1:
            # Add "color" dimension:
            currentImage = np.expand_dims(currentImage, axis=-1)

        # Show the input image:
        if displayImages:
            showImage("Input image [Pre-processed]", currentImage)

        # Into the list of samples:
        testDataset[currentClass].append(currentImage)
        # Image counter goes up +1:
        totalDatasetSamples += 1

# Get total samples in dataset:
print("[FaceNet Test] Dataset Samples:", totalDatasetSamples)
print("Loaded: [" + str(imagesRange) + "] images per class.")

# Create the positive pairs:
if pairsPerClass == -1:
    pairsPerClass = 0.5 * (imagesRange ** 2.0) - (0.5 * imagesRange) - 7e-12
    pairsPerClass = math.ceil(pairsPerClass)

print("Creating: " + str(pairsPerClass) + " pairs per class...")

# Positive pairs are stored here:
positivePairs = []

for currentClass in testDataset:
    # Get images of current class:
    classImages = testDataset[currentClass]

    # Shuffle images:
    random.shuffle(classImages)

    # Get total samples for this class:
    classSamples = len(classImages)

    # processed samples counter:
    processedPairs = 0

    for i in range(pairsPerClass):
        choices = list(range(0, classSamples))
        randomSamples = [0, 0]

        # Randomly choose a first sample:
        randomSamples[0] = random.choice(choices)

        # Randomly choose a second sample, excluding the one already
        # chosen:
        randomSamples[1] = random.choice(list(set(choices) - set([randomSamples[0]])))

        # Print the chosen samples:
        print("Processing pair: " + str(processedPairs), randomSamples)

        # Store sample pair here:
        tempList = []

        # Get images from dataset:
        for s in range(len(randomSamples)):
            sampleIndex = randomSamples[s]
            currentImage = classImages[sampleIndex]
            # Show image
            if displayImages:
                showImage(currentClass + " Pair: " + str(s), currentImage)

            # Into the temp list:
            tempList.append(currentImage)

        # Finally, store class code:
        classCode = classesDictionary[currentClass]
        tempList.append(classCode)
        # Into the positive pairs list:
        positivePairs.append(tempList)
        processedPairs += 1

    # Done creating positive pairs for this class:
    totalPairs = len(positivePairs)
    print("[" + currentClass + "]" + " Pairs Created: " + str(processedPairs) + ", Class: " + currentClass,
          "(" + str(classesDictionary[currentClass]) + ")", " Total: " + str(totalPairs))

# Shuffle the list of positive pairs:
random.shuffle(positivePairs)

# Create the "test batch":
testBatch = []  # List of pairs and real labels

# Compute the dataset portions:
totalPositives = int(positivePortion * datasetSize)
totalNegatives = datasetSize - totalPositives

print("Positive Samples: " + str(totalPositives) + " Negative Samples: " + str(totalNegatives) +
      " Total: " + str(totalPositives + totalNegatives))

# Get total number of pairs:
totalPairs = len(positivePairs)

# Randomly choose n positive examples from the pairs list:
choicesArray = np.arange(0, totalPairs, 1, dtype=int)
positiveSamples = np.random.choice(choicesArray, totalPositives, replace=True)

# Store the positive random pairs in the batch array:
print("Storing positive pairs for test batch...")
for i in range(totalPositives):

    # Get current pair of images:
    randomSample = positiveSamples[i]
    currentSample = positivePairs[randomSample]

    # Check images (if proper data type):
    if displayImages:
        showImage("[Positive] Sample 1", currentSample[0])
        showImage("[Positive] Sample 2", currentSample[1])

    # Into the batch - Pair and label (1):
    testBatch.append(([currentSample[0], currentSample[1]], 1))

positiveBatchSize = len(testBatch)
print("Positive pairs stored: " + str(positiveBatchSize))

# Store the negative random pairs:
print("Storing negative pairs for test batch...")
for i in range(totalNegatives):

    # Randomly generated negative sample row here:
    tempList = []
    pastClass = -1

    # Randomly choose two sample rows:
    for s in range(2):
        getRows = True
        while getRows:
            # Get a random row from the positive pairs list,
            # "Flatten" to an integer, since the return value is
            # an array:
            randomChoice = np.random.choice(choicesArray, 1)[0]

            # Actually Get the random row:
            randomRow = positivePairs[randomChoice]

            # Get the sample's class:
            rowClass = randomRow[-1]

            if rowClass != pastClass:
                # Randomly choose one of the two images:
                randomChoice = random.randint(0, 1)
                randomSample = randomRow[randomChoice]

                # Show the random sample:
                if displayImages:
                    showImage("randomSample: " + str(s), randomSample)

                # Into the temp list:
                tempList.append(randomSample)
                # Stop the loop:
                getRows = False

                # Present is now past:
                pastClass = rowClass

    # Got the two negative samples, thus the negative pair is ready:
    testBatch.append((tempList, 0))

negativeBatchSize = len(testBatch) - positiveBatchSize
print("Negative pairs stored: " + str(negativeBatchSize))

# Shuffle list:
random.shuffle(testBatch)
print("Total pair samples in batch: " + str(len(testBatch)))

# Build the faceNet model:

model = faceNet.build(height=imageHeight, width=imageWidth, depth=imageChannels, namesList=["image1", "image2"],
                      embeddingDim=embeddingSize, alpha=lr, epochs=trainingEpochs)

# Load in weights:
weightsFilePath = outputPath + weightsFilename
print("[INFO] -- Loading faceNet weights file from: " + weightsFilePath)
model.load_weights(weightsFilePath)

# Get summary:
model.summary()

yPred = []
yTest = []

# Test the batch:
for b in range(len(testBatch)):
    # Get current pair & label:
    testPair = testBatch[b][0]
    testLabel = testBatch[b][1]

    # Add "batch" dimension:
    image1 = np.expand_dims(testPair[0], axis=0)
    image2 = np.expand_dims(testPair[1], axis=0)

    # Pack into dict:
    tempDict = {"image1": image1, "image2": image2}

    # Send to CNN, get probability:
    predictions = model.predict(tempDict)

    # Get current prediction:
    currentPrediction = predictions[0][0]

    # Set label color:
    if currentPrediction < 0.5:
        predictedLabel = 0
        classText = "Negative"
        borderColor = (0, 0, 255)
    else:
        predictedLabel = 1
        borderColor = (0, 255, 0)
        classText = "Positive"

    # Check the info:
    text = str(testLabel) + " : " + f'{currentPrediction:.4f}'
    print(b, "Class: ", classText, " Predicted: ", f'{currentPrediction:.4f}', " Real: ", testLabel)

    # Store the info for CF plotting:
    yTest.append(testLabel)
    # Store the info for CF plotting:
    yPred.append(predictedLabel)

    # Compose pair image
    # Check the images:
    imageList = [testPair[0], testPair[1]]

    # Horizontally concatenate images:
    stackedImage = cv2.hconcat(imageList)
    imageDimensions = len(stackedImage.shape)

    if imageDimensions < 3:
        (height, width) = stackedImage.shape
        depth = 1
    else:
        (height, width, depth) = stackedImage.shape

    # Get image type:
    imageType = stackedImage.dtype

    # Check type and convert to uint8:
    if imageType != np.dtype("uint8"):
        stackedImage = stackedImage * 255.0
        stackedImage = stackedImage.clip(0, 255).astype(np.uint8)

        # Convert channels:
        if depth != 3:
            stackedImage = cv2.cvtColor(stackedImage, cv2.COLOR_GRAY2BGR)

    # Draw rectangle:
    cv2.rectangle(stackedImage, (0, 0), (width - 1, height - 1), borderColor, 1)

    # Resize image:
    imgHeight, imgWidth = stackedImage.shape[0:2]
    aspectRatio = imgHeight / imgWidth

    # Up-scaling using original aspect ratio:
    displaySize = 150
    newWidth = int(displaySize // aspectRatio)

    stackedImage = cv2.resize(stackedImage, (newWidth, displaySize), interpolation=cv2.INTER_LINEAR)

    # Create text strip:
    stripHeight = int(0.2 * displaySize)
    stripShape = (stripHeight, newWidth, 3)
    textStrip = np.zeros(shape=stripShape, dtype="uint8")

    # Text & color:
    if predictedLabel == testLabel:
        if testLabel == 1:
            textColor = (0, 255, 0)
        else:
            textColor = (0, 0, 255)
    else:
        text = text + " [x]"
        textColor = (0, 128, 255)

    cv2.putText(textStrip, text, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=textColor, thickness=1)
    # showImage("text strip", textStrip)

    # Vertically concatenate images:
    stackedImage = cv2.vconcat([stackedImage, textStrip])

    # Show the positive/negative pair of images:
    if showClassified:
        showImage("Test Sample", stackedImage)

# Compute precision & recall:
modelPrecision = precision_score(yTest, yPred)
modelRecall = recall_score(yTest, yPred)
print((modelPrecision, modelRecall))

# Compute confusion matrix:
result = confusion_matrix(yTest, yPred, normalize='pred')
print(result)
disp = ConfusionMatrixDisplay(confusion_matrix=result)
disp.plot()
plt.show()

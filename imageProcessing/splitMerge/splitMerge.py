# File        :   splitMerge.py
# Version     :   0.1.0
# Description :   [WIP] Brute-force implementation of Split Merge
#                 for image segmentation

# Date:       :   Apr 18, 2022
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT


import numpy as np
import cv2
import math
from datetime import date, datetime


# Reads image via OpenCV:
def readImage(imagePath):
    # Loads image:
    inputImage = cv2.imread(imagePath)
    # Checks if image was successfully loaded:
    if inputImage is None:
        print("readImage>> Error: Could not load Input image.")
    return inputImage


# Defines a re-sizable image window:
def showImage(imageName, inputImage):
    cv2.namedWindow(imageName, cv2.WINDOW_NORMAL)
    cv2.imshow(imageName, inputImage)
    cv2.waitKey(0)


# Writes an PGN image:
def writeImage(imagePath, inputImage):
    imagePath = imagePath + ".png"
    cv2.imwrite(imagePath, inputImage, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print("Wrote Image: " + imagePath)


# Produces out string for image writing:
def getOutString(currentDate):
    # Format date
    currentDate = currentDate.strftime("%b-%d-%Y")

    # Get time:
    currentTime = datetime.now()
    currentTime = currentTime.strftime("%H:%M:%S")

    # Drop them nasty ":s":
    dateString = "_" + currentTime[0:2] + "-" + currentTime[3:5] + "-" + currentTime[6:8]
    print("Current Time: " + currentTime + " Date String: " + dateString)
    dateString = currentDate + dateString

    return dateString

# Divides the image into 4 quads:
def divideImage(tempImage):
    # Get image dimensions (shape):
    (height, width) = tempImage.shape[:2]

    # Deep copy of the input:
    cropImage = tempImage.copy()

    # Set the quad coordinates for cropping:
    # X dim/axis:
    xStart = [0, 0.5 * width, 0, 0.5 * width]                   # x -> [q0, q1, q2, q3]
    xEnd = [width * 0.5, width * 0.5, width * 0.5, width * 0.5]

    # Y axis/dim:
    yStart = [0, 0, height * 0.5, height * 0.5]                 # y -> [q0, q1, q2, q3]
    yEnd = [height * 0.5, height * 0.5, height * 0.5, height * 0.5]

    # Store the quad coordinates (bounding rectangle)
    # and sub-images (quad images) here:
    outCoords = []
    outImages = []

    # Produce 4 quads:
    for i in range(4):

        # Get croppping data:
        rectX = int(xStart[i])
        rectY = int(yStart[i])
        rectWidth = rectX + int(xEnd[i])
        rectHeight = rectY + int(yEnd[i])

        # Slice/crop image:
        currentQuad = cropImage[rectY:rectHeight, rectX:rectWidth]
        # (h, w) = currentQuad.shape[:2]
        # print("coords 1: " + str(rectX) + ", " + str(rectY) + ", " + str(rectWidth) + ", "
        #     + str(rectHeight)+ " (W: "+str(w)+" H: "+str(h)+")")

        # Quad goes into the list:
        outImages.append((currentQuad))
        # currentDate = date.today()
        # sampleName = str(s) + "-" + getOutString(currentDate)
        # writeImage("D://opencvImages//sm//quad_"+str(i)+"_"+getOutString(currentDate), currentQuad)

        # Show some debug info, draw rectangles where
        # image has been sliced:
        # color = list(np.random.random(size=3) * 256)
        # cv2.rectangle(tempImage, (rectX, rectY), (rectWidth, rectHeight), color, 2)
        # showImage("Quads", tempImage)

        # Bounding rectangle goes into list:
        outCoords.append([rectX, rectY, rectWidth, rectHeight])

    # Return the data as a tuple:
    return (outImages, outCoords)


# Set the resources paths:
path = "D://opencvImages//"
fileName = "peaceHand.png"

# Read the image via OpenCV and show it:
inputImage = readImage(path + fileName)
showImage("Input Image", inputImage)

# Gaussian Blur:
sigma = (5, 5)
filteredImage = cv2.GaussianBlur(inputImage, sigma, 0)
showImage("filteredImage [Gaussian Blur]", filteredImage)

# To Lab:
hasvImage = cv2.cvtColor(filteredImage, cv2.COLOR_BGR2HSV)

# Get image dimensions:
(imageHeight, imageWidth) = hasvImage.shape[:2]
print("Image W x H: " + str(imageWidth) + ", " + str(imageHeight))
# The new image:
newImage = np.zeros((imageHeight, imageWidth, 3), np.uint8)
# Tree divisions are drawn here:
treeImage = newImage.copy()

# The quad tree "depth"
maxDepth = 3
# Size maxDepth by number of quads
numberOfQuads = 4

# This array stores the images:
quadTree = np.empty((numberOfQuads, maxDepth), np.matrix)
# This array stores the quad's bounding rectangles/rois:
quadCoords = np.empty((numberOfQuads, maxDepth), np.matrix)

# Quad "on/off" offsets -> quad: (x, y) for quads 0 - 3
dictOffsets = {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)}

# The local subvision counter:
localQuadCount = 0

# Level 0 of the tree, just the "parent quads":
# Get parents:
parents, parentsCoords = divideImage(hasvImage)


for p in range(len(parents)):

    # Parents are the first level of the tree:
    currentParent = parents[p]
    (h, w) = currentParent.shape[:2]
    # Store in the tree, as a matrix
    # of one element:
    quadTree[p][0] = [currentParent]
    # showImage("Current Parent", quadTree[p][0][0])

    # Get coords for this parent and store the quad
    # coordinates in the ROIS array:
    quadCoords[p][0] = np.array(parentsCoords[p])

    # Draw the parents rois:
    rectX = quadCoords[p][0][0]
    rectY = quadCoords[p][0][1]
    rectWidth = quadCoords[p][0][2]
    rectHeight = quadCoords[p][0][3]

    # Print the coordinates, make sure you are not
    # fucking this up:
    # print((rectX, rectY, rectWidth-rectX, rectHeight-rectY, w, h))

    # Draw parents quads in green:
    color = (0, 255, 0)
    cv2.rectangle(treeImage, (rectX, rectY), (rectWidth, rectHeight), color, 2)
    showImage("Tree Division", treeImage)

# # Check out parents:
for i in range(len(quadTree[:, 0])):

    currentParent = quadTree[i][0][0]
    showImage("Current Parent", currentParent)

# Traverse and construct the tree of max depth,
# starting at 1 coz level 0 are the parents:
for l in range(1, maxDepth):

    print("Current tree level: " + str(l))
    # Compute the division/tree current level,
    # Every iteration the tree is divided in half accross the two dimensions
    # (width and height):
    divisonLevel = (int(imageWidth / pow(2, l)), int(imageHeight / pow(2, l)))

    quadCount = 0
    offsetIndex = 0

    # Parent quad offset accumulation,
    # one per quad:
    dimAccum = [0, 0, 0, 0]

    # Branch length, which should
    # equal 4 quads:
    branchLength = len(quadTree[:, l])

    if (branchLength != numberOfQuads):
        print("Something's up with the branch length: " + str(branchLength))
    for q in range(4):

        # Get image from last level at this
        # position, as the tree is buit using the last level
        # information:
        currentQuad = quadTree[q][l - 1]

        # The
        dimAccum[0] = 0
        dimAccum[1] = 0

        print("Accumulations reset")

        # Check if there are images in this quad:
        if currentQuad is not None:

            # Get total images in this quad:
            totalImages = len(currentQuad)
            # print("totalImages in quad: " + str(totalImages))

            for i in range(totalImages):

                # Get current image:
                currentImage = quadTree[q][l - 1][i]
                s = "Level: " + str(l - 1) + " Quad: " + str(q) + " Image: " + str(i)
                showImage(s, currentImage)

                # Divide current image into 4 sub-quads:
                subQuads, subQuadCoords = divideImage(currentImage)
                # (th, tw)  = subQuads.shape[:2]


                # !!!!! BUG BUG BUG BUG BUG BUG BUG BUG !!!!!
                # !!!!! The images and coordinates are not being appended, they are being replaced !!!!

                # f = quadTree[q][l]
                # f1 = np.hstack((quadTree[q][l], subQuads))
                # quadTree[q][l] = np.vstack()
                # print(quadTree[q][l].shape)

                # Each subquad goes into this tree cell/node, at this level:
                quadTree[q][l] = subQuads

                totalSubQuads = len(quadTree[q][l])
                print("totalSubQuads created: " + str(totalSubQuads))
                print("Stored: " + str(totalSubQuads)+ " sub quads in quad: "+str(q)+", level: "+str(l))
                # Check out the current sub-quads:
                for j in range(totalSubQuads):

                    # Get sub quad:
                    currentSubQuad = quadTree[q][l][j]

                    # Get sub quad statistics:
                    quadMean, quadStdDev = cv2.meanStdDev(currentSubQuad)
                    print("L: " + str(quadStdDev[0]) + " A: " + str(quadStdDev[1]) + " B: " + str(quadStdDev[1]))
                    s = "SubQuad: " + str(j)

                    # If current std dev is below a threshold, it does not
                    # vary significantly, flood it with the mean:
                    stdDevThreshold = 40

                    # # Only check the H channel:
                    if (quadStdDev[0] < stdDevThreshold):
                        print("Flood Fill Subquad...")
                        # Set the Flood-fill color:
                        floodColor = (int(quadMean[0][0]), int(quadMean[1][0]), int(quadMean[2][0]))

                        # Set up Flood-fill:
                        fillThreshold = 50
                        loDiff = (fillThreshold, fillThreshold, fillThreshold)
                        upDiff = (fillThreshold, fillThreshold, fillThreshold)
                        cv2.floodFill(currentSubQuad, None, seedPoint=(0, 0), newVal=floodColor, loDiff=loDiff, upDiff=upDiff)

                        # Modified image goes into tree:
                        quadTree[q][l][j] = currentSubQuad

                    showImage(s, currentSubQuad)

                # !!!!! BUG BUG BUG BUG BUG BUG BUG BUG !!!!!
                # Get coordinates/bounding box for this subquad and
                # store em in the rois array:
                quadCoords[q][l] = np.array(subQuadCoords)
                totalRois = len(quadCoords[q][l])

                print("totalImages in quad: " + str(totalImages))
                print("totalRois received: " + str(totalSubQuads))
                print("Roi Quad: " + str(q) + " level: " + str(l))

                # Get total rois per level:
                levelRois = numberOfQuads * totalSubQuads * totalImages

                # Set the quad offsets:
                (quadHeight, quadWidth) = currentImage.shape[:2]

                # Each tree subdivision creates 4 images (or a quad),
                # this couunter keeps track of the number of quads/images
                # generated:
                imageCounter = 0

                # Get the on/off offsets for each of the four
                # images produced:
                curretOffsetX = dictOffsets[offsetIndex][0]
                curretOffsetY = dictOffsets[offsetIndex][1]

                print("Division Level -  x: " + str(divisonLevel[0]) + " y: " + str(divisonLevel[1]))
                print("Current Offset - (" + str(curretOffsetX) + " , " + str(curretOffsetY) + ") [" + str(
                    offsetIndex) + "]")

                # Get/compute the "Global" Offsets:
                # The global offsets are computed according to the current quad produced
                # and its position, and the position of its images, on the image:
                dimAccum[2] = dictOffsets[q][0] * (0.5 * imageWidth)
                dimAccum[3] = dictOffsets[q][1] * (0.5 * imageHeight)

                # For debug, draw the current starting point of every quad
                # division on each iteration (pink):
                point2 = (int(dimAccum[2]), int(dimAccum[3]))
                cv2.line(treeImage, point2, point2, (230, 30, 230), 5)
                showImage("Tree Division", treeImage)

                # Draw the subquad rois/bounding rectangles for
                # the current quad:
                for r in range(totalRois):

                    # Quad Number goes from 0 to >= 4:
                    quadNumber = math.floor((r) / numberOfQuads)

                    # Get current sub quad cooords:
                    currentCoords = quadCoords[q][l][r]

                    # Set the individual dimensions:
                    rectX = currentCoords[0]
                    rectY = currentCoords[1]
                    rectWidth = currentCoords[2]
                    rectHeight = currentCoords[3]

                    # Compute the offset coordinates of every subdivision
                    # produced so far:
                    xOffset = dimAccum[2] + curretOffsetX * dimAccum[0] + dictOffsets[quadNumber][0] *  quadWidth
                    yOffset = dimAccum[3] + curretOffsetY * dimAccum[1] + dictOffsets[quadNumber][1] *  quadHeight

                    # print("Roi Coords: " + str(rectX) + ", " + str(rectY) + ", " + str(rectWidth - rectX) + ", "
                    #      + str(rectHeight - rectY))

                    # Set some random color:
                    color = list(np.random.random(size=3) * 256)
                    # Draw the roi rectangle:
                    cv2.rectangle(treeImage, (int(xOffset + rectX), int(yOffset + rectY)),
                                  (int(xOffset + rectWidth), int(yOffset + rectHeight)), color, 2)
                    showImage("Tree Division", treeImage)

                    # Paste every quad subdivision on the
                    # new image:
                    pasteX = int(xOffset + rectX)
                    pasteY = int(yOffset + rectY)
                    pasteWidth = int(xOffset + rectWidth)
                    pasteHeight = int(yOffset + rectHeight)

                    pasteImage = quadTree[q][l][r]
                    showImage("pasteImage", pasteImage)
                    newImage[pasteY:pasteHeight, pasteX:pasteWidth] = quadTree[q][l][r]
                    showImage("New Image", newImage)

                    # Quad counter goes up by 1:
                    imageCounter += 1

                    # Every quad this check is evaluated...
                    # Adds the local offsets produced so far in every quad:

                    if (imageCounter == 4):
                        print("Got a Quad (4 Images): " + str(offsetIndex))

                        # Local quad count:
                        if (localQuadCount == 4):
                            localQuadCount = 1
                        else:
                            localQuadCount += 1

                        print("Parent Quad: "+str(q)+" localQuadCount: " + str(localQuadCount))

                        # Local/child quads accumulations:
                        if (localQuadCount == 1):
                            dimAccum[0] = dimAccum[0] + rectWidth
                            dimAccum[1] = 0
                        elif (localQuadCount == 2):
                            dimAccum[0] = 0
                            dimAccum[1] = dimAccum[1] + rectHeight
                        elif(localQuadCount == 3):
                            dimAccum[0] = dimAccum[0] + rectWidth

                        # For debug, draw the next starting point of every quad
                        # division on each iteration (red):
                        point = (dimAccum[0], dimAccum[1])
                        cv2.line(treeImage, point, point, (0, 0, 255), 10)
                        showImage("Tree Division", treeImage)

                        # Index that controls the offset
                        # hash table:
                        if (offsetIndex == 3):
                            offsetIndex = 0
                        else:
                            offsetIndex += 1

            print("Done processing level: " + str(l - 1) + " Quad: " + str(q) + " Image: " + str(i))

        else:
            print("[ Got None in Level: " + str(l - 1) + ", Quad: " + str(q) + " ]")

# To BGR:
bgrImage = cv2.cvtColor(newImage, cv2.COLOR_HSV2BGR)
showImage("Split BGR", bgrImage)

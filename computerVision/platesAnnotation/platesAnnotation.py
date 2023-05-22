import tkinter as tk
from tkinter import messagebox
import PIL.Image, PIL.ImageTk
import os
import random

import cv2
import numpy as np


# Defines a re-sizable image window:
def showImage(imageName, inputImage):
    cv2.namedWindow(imageName, cv2.WINDOW_NORMAL)
    cv2.imshow(imageName, inputImage)
    # cv2.waitKey(0)


# Writes an PGN image:
def writeImage(imagePath, inputImage):
    imagePath = imagePath + ".png"
    cv2.imwrite(imagePath, inputImage, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print("Wrote Image: " + imagePath)


def setStripInfo(text, location, color):
    global stripImg

    # Set text:
    font = cv2.FONT_HERSHEY_SIMPLEX

    x = location[0]
    y = location[1]

    # Draw mouse coordinates on new image:
    cv2.putText(stripImg, text, (x, y), font, 0.5, color, 1, cv2.LINE_8)


def makeStrip(stripWidth, stripHeight=50):
    # Create the image:
    img = np.zeros([stripHeight, stripWidth, 3], dtype=np.uint8)

    return img


def setImagePath(path, name):
    global imagePath
    imagePath = path
    global imageName
    imageName = name
    print("Image Name set: " + imagePath + imageName)


def getImage(*args):
    global imagePath
    global imageName
    imgStr = imagePath + imageName
    print("Reading image: " + imgStr)
    img = cv2.imread(imgStr)

    return img


def reloadImage(*args):
    try:
        print("Reloading Image...")
        global inputImage

        # Get image from path:
        inputImage = getImage()

        # Create deep copy:
        global inputImageCopy
        inputImageCopy = inputImage.copy()

        # Clear the rectified image:
        global rectifiedImage
        if rectifiedImage.size != 0:
            rectifiedImage = np.array([])

        # Re-create the info strip:
        # Get image dimensions:
        h, w, _ = inputImage.shape

        global stripImg
        # Create the info strip
        stripImg = makeStrip(w)

        # Vertical stack of the images:
        completeImg = np.vstack((inputImage, stripImg))

        # Show Image:
        cv2.imshow("image", completeImg)
        print("Image reloaded.")
    except ValueError:
        pass


def unwarpPerspective(*args):
    try:
        print("Unwarping Image...")
        # Get rectified aspect ratio:
        global rectifiedAspectRatio

        # Get un modified image from path:
        global rectifiedImage
        originalImage = getImage()

        # Reshape the array:
        pts = np.zeros((4, 2), dtype="float32")

        pts[0] = perspectivePoints[0]
        pts[1] = perspectivePoints[1]
        pts[2] = perspectivePoints[2]
        pts[3] = perspectivePoints[3]

        rect = np.zeros((4, 2), dtype="float32")

        # the top-left point has the smallest sum whereas the
        # bottom-right has the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # compute the difference between the points -- the top-right
        # will have the minumum difference and the bottom-left will
        # have the maximum difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # now that we have our rectangle of points, let's compute
        # the width of our new image
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

        # ...and now for the height of our new image
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

        # take the maximum of the width and height values to reach
        # our final dimensions
        maxWidth = max(int(widthA), int(widthB))
        maxHeight = max(int(heightA), int(heightB))

        # construct our destination points which will be used to
        # map the screen to a top-down, "birds eye" view
        dst = np.array([
            [0, 0], [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
            dtype="float32")

        # Add the whole image offset:
        dst[0][0] = dst[0][0] + tl[0]
        dst[0][1] = dst[0][1] + tl[1]

        dst[1][0] = dst[1][0] + tl[0]
        dst[1][1] = dst[1][1] + tl[1]

        dst[2][0] = dst[2][0] + tl[0]
        dst[2][1] = dst[2][1] + tl[1]

        dst[3][0] = dst[3][0] + tl[0]
        dst[3][1] = dst[3][1] + tl[1]

        # calculate the perspective transform matrix and warp
        # the perspective to grab the screen:
        M = cv2.getPerspectiveTransform(rect, dst)

        global inputImage
        global inputImageCopy

        inputImage = cv2.warpPerspective(originalImage, M, (originalImage.shape[1], originalImage.shape[0]))
        inputImageCopy = inputImage.copy()

        global rectifiedImage
        rectifiedImage = inputImage

    except ValueError:
        pass


# mouse callback function
def mouseHandler(event, x, y, flags, param):
    global ix, iy

    font = cv2.FONT_HERSHEY_SIMPLEX
    global inputImage
    global inputImageCopy

    # Retrieves fresh, new drawing on every call (the img copy):
    inputImage = inputImageCopy.copy()

    # Draw mouse coordinates on new image:
    cv2.putText(inputImage, "(" + str(x) + "," + str(y) + ")", (x, y), font,
                0.5, (255, 0, 0), 1, cv2.LINE_8)

    # Stores the "state" of the current processing chain:
    global cropProcessing
    global perspProcessing

    # Triggered at mouse left  click:
    if event == cv2.EVENT_LBUTTONDOWN:
        # Show cursor coordinates:
        print("x: " + str(x), "y: " + str(y))
        # Draw nice circles at clicked point:
        cv2.circle(inputImage, (x, y), 5, (255, 0, 0), -1)

        # Set the static copy with the new info (the circle):
        inputImageCopy = inputImage

        # Store the crop points at location
        # described by the "state"
        # -> 0 - first points
        # -> 1 - second points:
        cropPoints[cropProcessing] = (x, y)

        # Increase processing "state":
        cropProcessing = cropProcessing + 1

        # Store mouse global coordinates:
        ix, iy = x, y

    # Draw rectangle as long as state is not 0:
    if cropProcessing != 0:

        # Get starting point:
        rx = cropPoints[0][0]
        ry = cropPoints[0][1]
        # Get rectangle width and height:
        w = x - rx
        h = y - ry
        # Draw rectangle in real-time:
        # print("rx: " + str(rx), "ry: " + str(ry) + " w: " + str(w) + " h:" + str(h))
        cv2.rectangle(inputImage, (int(rx), int(ry)), (int(x), int(y)), rectColor, 1)

        # Second click, after first click:
        if cropProcessing == 2:
            # Set the static copy with the new info (the circle + the rectangle):
            inputImageCopy = inputImage
            # Reset state:
            cropProcessing = 0
            # Save rect coordinates:
            lastRectangleCoordinates[0] = cropPoints[0][0]
            lastRectangleCoordinates[1] = cropPoints[0][1]
            lastRectangleCoordinates[2] = cropPoints[1][0]
            lastRectangleCoordinates[3] = cropPoints[1][1]
            # Crop:
            cropImage()
            # Set Focus:
            root.focus_force()

    # Triggered at mouse right click:
    if event == cv2.EVENT_RBUTTONDOWN:
        # Show cursor coordinates:
        print("x: " + str(x), "y: " + str(y))

        # Draw nice circles at clicked point:

        cv2.circle(inputImage, (x, y), 5, (0, 0, 255), -1)
        # showImage("Shit", inputImage)

        # Set the static copy with the new info (the circle):
        inputImageCopy = inputImage

        # Store the crop points at location
        # described by the "state"
        # -> 0 - first points
        # -> 1 - second points:
        perspectivePoints[perspProcessing] = (x, y)

        # Increase processing "state":
        perspProcessing = perspProcessing + 1

        # Store mouse global coordinates:
        ix, iy = x, y

    # Draw lines as long as state is not 0:
    if perspProcessing != 0:

        # Get starting point:
        lsx = perspectivePoints[perspProcessing - 1][0]
        lsy = perspectivePoints[perspProcessing - 1][1]

        # Get last point
        lex = x
        ley = y
        # Draw line in real-time:
        cv2.line(inputImage, (int(lsx), int(lsy)), (int(lex), int(ley)), (0, 0, 255), 2)

        # Process the 4 points:
        if perspProcessing == 4:
            print("Got 4 points...")

            inputImageCopy = inputImage
            # Reset state:
            perspProcessing = 0

            # Unwarp perspective:
            unwarpPerspective()

            # Set Focus:
            root.focus_force()

    # Show them values:
    # print("Rect Process: " + str(cropProcessing))

    # Set the info to strip
    currentImageString = sampleList[imageCounter]
    setStripInfo("Image: " + str(currentImageString) + " (" + str(imageCounter + 1) + "/" + str(totalSamples) + ")",
                 (5, 15), (0, 255, 0))
    setStripInfo(processedBlobInfo, (5, 35), (0, 255, 255))
    # Vertical stack of the images:
    completeImg = np.vstack((inputImage, stripImg))

    cv2.imshow("image", completeImg)


def cropImage(*args):
    try:
        print("Cropping Image...")
        # Get un modified image from path:
        global croppedImage
        if rectifiedImage.size != 0:
            originalImage = rectifiedImage
        else:
            originalImage = getImage()

        # Get crop coordinates from list:
        x = cropPoints[0][0]
        y = cropPoints[0][1]
        w = cropPoints[1][0]
        h = cropPoints[1][1]

        # crop Image:
        croppedImage = originalImage[y:h, x:w]
        cv2.imshow("Current Crop", croppedImage)
        print("Cropped at: x - " + str(x) + ", y - " + str(y) + " w: " + str(w) + " h: " + str(h))

    except ValueError:
        pass


def processImage(*args):
    print("Cropping Image...")

    # Crop Image
    global croppedImage

    # originalImage = cv2.imread(imagePath + imageName)
    #
    # # Get croppin rectangle:
    # x = cropPoints[0][0]
    # y = cropPoints[0][1]
    # w = cropPoints[1][0]
    # h = cropPoints[1][1]
    #
    # croppedImage = originalImage[y:h, x:w]
    # cv2.imshow("Current Crop", croppedImage)
    # print("Cropped at: x - " + str(x) + ", y - " + str(y) + " w: " + str(w) + " h: " + str(h))
    #

    # Resize the image:
    # Get original aspect ratio:
    (h, w) = croppedImage.shape[:2]
    originalAspectRatio = w / h

    global resizeHeight
    global resizeWidth
    # New width:
    resizeWidth = resizeHeight * originalAspectRatio

    # To int:
    resizeWidth = int(resizeWidth)
    resizeHeight = int(resizeHeight)

    print("Resizing image to w: " + str(resizeWidth) + " h: " + str(resizeHeight))

    newSize = (int(resizeWidth), int(resizeHeight))
    # Interpolation method:
    interpolation = cv2.INTER_LINEAR
    # Resize image:
    croppedImage = cv2.resize(croppedImage, newSize, None, None, None, interpolation)
    # cv2.imshow("Resized Image", croppedImage)
    # messagebox.showinfo('Fuck you', 'Fuck you')
    # Set focus to main window:
    # root.focus_force()
    # print("Waiting...")
    # cv2.waitKey()
    # retval = cv2.waitKey()
    # print("retVal: "+str(retval))


def keyPressed(event):
    # Get key pressed
    currentKey = str(event.char).capitalize()
    print("Key Pressed: " + currentKey)
    w = tk.Label(root, text=currentKey)
    w.place(x=60, y=50)

    processImage()

    # Create out dir:
    outPath = os.path.join(outputPath, currentKey)
    isDir = os.path.isdir(outPath)
    totalFiles = 0

    if not isDir:
        os.mkdir(outPath)
        print("Created Directory: " + outPath)
        sampleCounter = 0
    else:
        print("Found Directory: " + outPath)
        # list of files:
        filesList = os.listdir(outPath)
        print(filesList)
        totalFiles = len(filesList)
        print("totalFiles: " + str(totalFiles))
        # Sample name:
        sampleCounter = totalFiles

    # Write image:
    sampleCounter += 1
    outImagePath = outPath + "//" + str(sampleCounter)
    writeImage(outImagePath, croppedImage)

    # Captured blob confirmation:
    # Get starting point:
    sx = lastRectangleCoordinates[0]
    sy = lastRectangleCoordinates[1]
    ex = lastRectangleCoordinates[2]
    ey = lastRectangleCoordinates[3]

    # Draw rectangle in real-time:
    cv2.rectangle(inputImageCopy, (int(sx), int(sy)), (int(ex), int(ey)), (0, 0, 255), 1)

    # Clean strip:
    infoY = 20
    clearHeight = stripImg.shape[0]
    cv2.rectangle(stripImg, (0, infoY), (int(250), int(clearHeight)), (0, 0, 0), -1)

    # Set info to info strip:
    global processedBlobInfo
    processedBlobInfo = "[" + currentKey + "] " + str(sampleCounter) + "/" + str(totalFiles + 1)


def getImageName(imageNumber):
    if imageNumber < 10:
        imageStringName = "0" + str(imageNumber)
    else:
        imageStringName = str(imageNumber)

    return imageStringName


def setNextSample(*args):
    global imagePath
    global imageName
    global imageCounter

    print("Loading new image...")
    print("imageCounter: " + str(imageCounter))

    if imageCounter < totalSamples - 1:
        imageCounter += 1
        currentSample = sampleList[imageCounter]
        imageName = rawDir + currentSample
        print("Next Image: " + imageName)
        reloadImage()
    else:
        print("Already got to last image.")


def setPrevSample(*args):
    global imagePath
    global imageName
    global imageCounter

    print("Loading new image...")
    print("imageCounter: " + str(imageCounter))

    if imageCounter > 0:
        imageCounter -= 1
        currentSample = sampleList[imageCounter]
        imageName = rawDir + currentSample
        print("Prev Image: " + imageName)
        reloadImage()
    else:
        print("Already got to first image.")


def createDataset(mode):
    print("Creating Data Set: "+mode)

    # Set number of samples per class:
    trainSamples = 30
    if mode == "Train":
        numberOfSamples = trainSamples
        #sampleRange = range(numberOfSamples)
    else:
        numberOfSamples = 10
        #random.randrange(1, 10)
        #sampleRange = range(numberOfSamples)

    # Get number of classes:
    classList = os.listdir(imagePath + samplesDir)
    print(classList)
    numberOfClasses = len(classList)
    print("Number of Classes: " + str(numberOfClasses))

    # Cell dimensions (with border):
    cellWidth = 70
    cellHeight = 100
    cellBorderColor = (175, 175, 175)

    # Create canvas, include gray borders:
    canvasHeight = numberOfClasses * cellHeight
    canvasWidth = numberOfSamples * cellWidth

    global dataSetImg
    dataSetImg = np.zeros([canvasHeight, canvasWidth, 3], dtype=np.uint8)

    # showImage("Data Set", dataSetImg)

    # Loop thrugh the images:
    for c in range(numberOfClasses):  # numberOfClasses

        currentClassName = classList[c]
        print(currentClassName)

        for s in range(numberOfSamples):  # numberOfSamples

            if mode == "Test":

                searchPath = imagePath + samplesDir + currentClassName
                fileList = os.listdir(searchPath)
                fileList = len(fileList)

                # random.seed(785)
                random.seed(); random.seed(); random.seed();

                if fileList > (trainSamples + 5):
                    random.seed(); random.seed();
                    randomNumber = random.randrange(trainSamples-1, fileList + 1)
                else:
                    random.seed(); random.seed();
                    randomNumber = random.randrange(1, trainSamples + 1)

                filePath = imagePath + samplesDir + currentClassName + "//" + str(randomNumber) + ".png"
                print("Class: " + str(currentClassName) + " Sample: " + str(randomNumber))

            else:
                filePath = imagePath + samplesDir + currentClassName + "//" + str(s + 1) + ".png"
                print("Class: " + str(currentClassName) + " Sample: " + str(s + 1))

            print(filePath)

            # Get image
            currentSample = cv2.imread(filePath)
            (sampleHeight, sampleWidth) = currentSample.shape[:2]

            # Add border to sample:
            cv2.rectangle(currentSample, (int(0), int(0)), (int(sampleWidth - 1), int(sampleHeight - 1)),
                          cellBorderColor, 1)

            # Create cell canvas:
            cellCanvas = np.zeros([cellHeight, cellWidth, 3], dtype=np.uint8)

            # Add border:
            cv2.rectangle(cellCanvas, (int(0), int(0)), (int(cellWidth - 1), int(cellHeight - 1)), cellBorderColor, 1)

            # Center sample into cell, compute offsets:
            oy = int(0.5 * (cellHeight - sampleHeight))
            ox = int(0.5 * (cellWidth - sampleWidth))

            # Paste into cell:
            cellCanvas[oy:oy + sampleHeight, ox:ox + sampleWidth] = currentSample
            # showImage("cell", cellCanvas)

            # Compute starting coordinates for this col, row:
            x = cellWidth * s
            y = cellHeight * c

            # Paste into dataset:
            dataSetImg[y:y + cellHeight, x:x + cellWidth] = cellCanvas

            # showImage("Dataset", dataSetImg)
            # writeImage(imagePath+outDir+"fuck", dataSetImg)

    print("Generated dataset.")
    if mode == "Train":
        writeImage(imagePath + outDir + "dataset-train", dataSetImg)
    else:
        writeImage(imagePath + outDir + "dataset-test", dataSetImg)


def createDataSetTrain(*args):
    createDataset("Train")


def createDataSetTest(*args):
    createDataset("Test")

# Main directory path:
imagePath = "D://opencvImages//plates//samples//"

# directory of raw samples:
rawDir = "raw//"
# directory of classified samples:
samplesDir = "classes//"
# Output directory:
outDir = "out//"

# list of files:
sampleList = os.listdir(imagePath + rawDir)
print(sampleList)
totalSamples = len(sampleList)

print("Total Samples on Dir: " + str(totalSamples))

for f in range(totalSamples):
    currentSampleName = sampleList[f]
    # print("CurrentSampleName: " + currentSampleName)

# Set image path and image name:
imageCounter = 0
imageName = rawDir + sampleList[0]

# Get image from path:
inputImage = getImage()
inputImageCopy = inputImage.copy()

# Get image dimensions:
imageHeight, imageWidth, _ = inputImage.shape

# Create the info strip
stripImg = makeStrip(imageWidth)

# Vertical stack of the images:
completeImg = np.vstack((inputImage, stripImg))

# Crop Points list:
cropPoints = [None] * 4

# Perspective Points:
perspectivePoints = [None] * 4

# Crop processing state variable:
cropProcessing = 0

# Perspective unwarp state variable:
perspProcessing = 0

# Set default color for rectangle:
rectColor = (0, 255, 0)
lastRectangleCoordinates = [None] * 4
processedBlobInfo = "-"

# Cropped image:
croppedImage = np.array([])

# Rectified image:
rectifiedImage = np.array([])

# Data set:
dataSetImg = np.array([])

# Rectified parameters (width, height):
rectifiedAspectRatio = 427 / 863
rectifiedHeight = 555

# Resizing parameters:
resizeHeight = 96
resizeWidth = -1

# Get directories at path:
outputPath = imagePath + samplesDir

# Mouse coordinates:
ix, iy = -1, -1

# Create the image window:
cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("image", mouseHandler)
cv2.imshow("image", completeImg)

# Create buttons window:
root = tk.Tk()
root.title("Blob Annotation Tool")
root.geometry('400x165+55+70')

# get keypress:
root.bind("<Key>", keyPressed)

# canvas for image:
# canvas = Canvas(root, width=imageWidth, height=imageHeight + buttonOfsset)
# canvas.pack()
# Convert numpy array image to tk image:
# photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(inputImage))
# Add image to canvas:
# @ canvas.create_image(0, buttonOfsset, image=photo, anchor=NW)

# mainframe = tk.Frame(root, padding="3 3 12 12")
reloadImageButton = tk.Button(root, text="Reload Img", command=reloadImage).place(x=10, y=10)
prevSampleButton = tk.Button(root, text="<", command=setPrevSample).place(x=100, y=10)
nextSampleButton = tk.Button(root, text=">", command=setNextSample).place(x=130, y=10)
dataSetTrainButton = tk.Button(root, text="Create Train Dataset", command=createDataSetTrain).place(x=10, y=100)
dataSetTestButton = tk.Button(root, text="Create Test Dataset", command=createDataSetTest).place(x=150, y=100)
# D = tk.Button(root, text="Process", command=processImage).place(x=200, y=0)

tk.Label(root, text="Blob Class: ").place(x=10, y=50)

# mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
# root.columnconfigure(0, weight=1)
# root.rowconfigure(0, weight=1)
#
# feet = StringVar()
# feet_entry = tk.Entry(mainframe, width=7, textvariable=feet)
# feet_entry.grid(column=2, row=1, sticky=(W, E))
#
# meters = StringVar()
# tk.Label(mainframe, textvariable=meters).grid(column=2, row=2, sticky=(W, E))
#
# B = tk.Button(root, text="Calculate", command=calculate)
# B.pack()
#
# tk.Label(mainframe, text="feet").grid(column=3, row=1, sticky=W)
# tk.Label(mainframe, text="is equivalent to").grid(column=1, row=2, sticky=E)
# tk.Label(mainframe, text="meters").grid(column=3, row=2, sticky=W)
#
# for child in mainframe.winfo_children():
#     child.grid_configure(padx=5, pady=5)
#
# feet_entry.focus()
# root.bind("<Return>", calculate)

root.mainloop()

# File        :   imageScrapper.py
# Version     :   0.8.0

# Description :   General purpose image scrapper with the following functions:

#                 1 - Renames image files ascendingly
#                 2 - Scraps a links list for images
#                 3 - Removes image duplicates via hashing

# Date:       :   Jun 19, 2023
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT

import re
import requests
from bs4 import BeautifulSoup, SoupStrainer

import cv2
import os

from functools import reduce
import numpy as np
import time


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

# Conversts a 32-bit numpy array hash to a hexadecimal
# string
def getHashString(inputArray):
    outString = ""
    for i in range(inputArray.shape[0]):
        # Get int from array, convert to hex and then to string:
        hexChar = str(hex(inputArray[i]))
        # Discard the prefix:
        hexChar = hexChar[2:]
        # Concatenate:
        outString = outString + str(hexChar)[-2:]
    # Done:
    return outString

# Natturally (alphanumerically) sorts an array of letters
# and numbers:
def naturalSort(inputList, prefix, imgExt=".png"):

    # Attempting to naturally sort the list of strings:
    tempList = []
    filenameCounter = 0
    prefixLen = len(prefix)
    for name in inputList:
        # Get the number from string:
        tempString = name[prefixLen:]
        tempString = tempString.split(".")
        fileNumber = int(tempString[0])
        # Store in temp list:
        tempList.append(fileNumber)

    # Sort the list (ascending)
    tempList.sort()
    # Prepare the (ordered) output list of strings:
    outList = [prefix + str(n) + imgExt for n in tempList]

    return outList

# Set the target sites (as a list:)
targetSites = ["https://www.bing.com/images/search?q=bob+odenkirk&form=HDRSC3&first=1"]

# Set output directory name:
dirName = "Bob Odenkirk"

coolDownTime = 1.0

renameImages = True
downloadImages = False
showDownloaded = False
saveImages = True
checkDownload = True

maxRetries = 5

removeDuplicates = True
showDuplicates = True
deleteFiles = True

imageScale = (0.3, 0.3)

projectPath = "D://dataSets//faces//"
outPath = projectPath + "celebrities//"
filenamePrefix = "image_"
imageExtension = ".png"

# Check directory:
outDirectory = outPath + dirName + "//"
print("Checking directory: " + outDirectory)

# Roll the pseudo-dice:
np.random.seed(69)

# Create output directory:
directoryExists = os.path.isdir(outDirectory)
if not directoryExists:
    print("Creating Directory: " + outDirectory)
    os.mkdir(outDirectory)
    directoryExists = os.path.isdir(outDirectory)
    if directoryExists:
        print("Successfully created directory: " + outDirectory)
else:
    print("Directory Found.")

# Check total files in directory:
_, _, files = next(os.walk(outDirectory))
fileCount = len(files)

print("Total Files in dir: " + str(fileCount))

# Image counter is number of files found:
imageCounter = fileCount

# Rename images:
if renameImages:
    renameCounter = 0
    for i, filename in enumerate(files):
        print(filename)

        # Set new file name:
        newFilename = filenamePrefix + str(renameCounter) + imageExtension

        # Full paths:
        oldPath = outDirectory + filename
        newPath = outDirectory + newFilename

        # Check if current file has a "valid" name:
        if oldPath != newPath:

            # Check the target is unique!
            alreadyExists = os.path.exists(newPath)
            while alreadyExists:
                # Set new file name:
                renameCounter += 1
                newFilename = filenamePrefix + str(renameCounter) + imageExtension
                newPath = outDirectory + newFilename
                alreadyExists = os.path.exists(newPath)

            # Rename:
            print("Renaming: ", oldPath, " to: ", newPath)
            os.rename(oldPath, newPath)
            # Update list:
            files[i] = newFilename

        renameCounter += 1

if downloadImages:

    totalDownloaded = 0
    pageExtensions = ["html", "htm"]

    for s, currentSite in enumerate(targetSites):
        print("[!] Attempting to download image from: " + currentSite)

        # Get base site (for image link generation)
        # Get main url, search for ".com":
        stringIndex = currentSite.find(".com")

        # Main url is from 0 to ".com" index + 4 (String range is exclusive):
        baseSite = currentSite[0:stringIndex + 4]
        print("Scrapping from:", baseSite, "(" + currentSite + ")")

        # Store download images for each site here:
        downloadedImages = {}
        # Store failed downloads here:
        failedImages = []

        # Set initial site for crawling:
        targetSite = currentSite
        galleryCounter = 0

        navigateGallery = True

        while navigateGallery:

            print("[!] Checking site: ", targetSite, " Gallery: ", galleryCounter)

            # Check site status:
            siteResponse = requests.get(targetSite)
            requestCode = siteResponse.status_code
            print("siteResponse Code", requestCode)

            if requestCode != 200:
                if requestCode == 404:
                    print("[-> Link not found]")
                    break
                elif requestCode == 403:
                    print("[-> Forbidden]")
                    break

            # Site Ok, proceed as usual:
            content = requests.get(targetSite).content
            soup = BeautifulSoup(content, "html.parser")

            # Look for gallery links:
            targets = soup.find_all("a", href=re.compile("photo"))

            # Store image links here:
            links = []

            for i, link in enumerate(targets):
                # Loop for link/tag attribute:
                if link.has_attr("href"):
                    if link["href"] != "#":
                        pageLink = link["href"]
                        print("Got an image link: ", pageLink)
                        links.append(pageLink)

            # Total images:
            totalLinks = len(links)
            print("[!] Total links: " + str(totalLinks))

            # Look for navigation links:
            navLinks = soup.find_all("a", class_=re.compile("link"), string=":: next ::")

            print(navLinks)

            # Total nav links:
            totalNavLinks = len(navLinks)
            print("[!] Total Nav Links: " + str(totalNavLinks))

            # Set next page from same gallery:
            if totalNavLinks > 0:
                tempLink = navLinks[0]
                tempLink = tempLink["href"]
                targetSite = currentSite + tempLink
                print("[!] Next link in gallery: " + targetSite)
                galleryCounter += 1
            else:
                print("[!] No more gallery pages...")
                navigateGallery = False

            # Downloaded images counter:
            downloadedCount = 0

            # Look for images to download:
            for i, link in enumerate(links):
                # Build full link:
                fullLink = baseSite + link
                print(i, "Processing link: " + fullLink)

                # Look for the image in this link:
                r = requests.get(fullLink)
                soup = BeautifulSoup(r.text, "html.parser")

                # Look for attributes of interest on target link:
                images = soup.find_all("img", src=re.compile("full"))

                # There might be more than one image at the link:
                for image in images:
                    # Cool down:
                    time.sleep(coolDownTime)

                    # Get image link via the "src" attribute:
                    imageLink = image["src"]
                    print("Image SRC:", imageLink)

                    # Get image name:
                    stringList = imageLink.split("?")
                    stringList = stringList[0].split("/")
                    imageName = stringList[-1]

                    # Remove ".extension":
                    imageName = imageName.split(".")
                    fileExtension = imageName[1]

                    # Check extension:
                    if fileExtension == "gif":
                        print("Got gif, skipping to next image...")
                        continue

                    # Set image name:
                    imageName = imageName[0]
                    print("ImageName:", imageName)

                    if saveImages:
                        # Check if image has been already downloaded...
                        if imageName not in downloadedImages:
                            keepDownloading = True
                            retryCounter = 0
                            while keepDownloading:

                                # Check link response:
                                linkResponse = requests.get(imageLink)
                                requestCode = linkResponse.status_code
                                print("requestCode", requestCode)

                                if requestCode != 200:
                                    # Store link for later new attempt:
                                    print("Storing: ", imageLink)
                                    failedImages.append(imageLink)
                                    if requestCode == 404:
                                        print("[-> Link not found]")
                                        break
                                    elif requestCode == 403:
                                        print("[-> Forbidden]")
                                        break

                                # Download image:
                                rawImage = requests.get(imageLink).content
                                fileOutput = outDirectory + filenamePrefix + str(imageCounter) + imageExtension

                                with open(fileOutput, "wb+") as f:
                                    f.write(rawImage)
                                    print("Wrote: " + fileOutput)

                                    if checkDownload:
                                        print("Checking image integrity...")
                                        # Load the image:
                                        currentImage = cv2.imread(fileOutput)
                                        # Check integrity:
                                        if currentImage is not None:
                                            if showDownloaded:
                                                showImage("Downloaded Image", currentImage)
                                            # Increase image counter, exit loop:
                                            imageCounter += 1
                                            # Increase download counter:
                                            downloadedCount += 1
                                            # Record download:
                                            downloadedImages[imageName] = True
                                            # Exit:
                                            keepDownloading = False
                                        else:
                                            if retryCounter < maxRetries:
                                                retryCounter += 1
                                                print("Download failed, retrying...", retryCounter)
                                                time.sleep(1.0)
                                            else:
                                                keepDownloading = False
                                                print("Download failed. Skipping image.")
                        else:
                            print("Already downloaded... Skipping download. [" + imageName + "]")

                print(downloadedImages)
                # Get effective count of downloaded images:
                print("Link: ", s, "Downloaded Images: " + str(downloadedCount))
                # Keep track of the amount of images downloaded:
                totalDownloaded = len(downloadedImages)

        totalFailedImages = len(failedImages)
        print("[!] Attempting to re-download failed images. Total: ", totalFailedImages)
        for failedLink in failedImages:

            print("Retrying: ", failedLink)

            keepDownloading = True
            retryCounter = 0
            downloadedCount = 0

            while keepDownloading:

                # Check link response:
                linkResponse = requests.get(failedLink)
                requestCode = linkResponse.status_code
                print("requestCode", requestCode)

                if requestCode != 200:
                    if requestCode == 404:
                        print("[-> Link not found]")
                        break
                    elif requestCode == 403:
                        print("[-> Forbidden]")
                        break

                # Download image:
                rawImage = requests.get(failedLink).content
                fileOutput = outDirectory + filenamePrefix + str(imageCounter) + imageExtension

                with open(fileOutput, "wb+") as f:
                    f.write(rawImage)
                    print("Wrote: " + fileOutput)

                    if checkDownload:
                        print("Checking image integrity...")
                        # Load the image:
                        currentImage = cv2.imread(fileOutput)
                        # Check integrity:
                        if currentImage is not None:
                            if showDownloaded:
                                showImage("Downloaded Image", currentImage)
                            # Increase image counter, exit loop:
                            imageCounter += 1
                            # Increase download counter:
                            downloadedCount += 1
                            # Exit:
                            keepDownloading = False
                        else:
                            if retryCounter < maxRetries:
                                retryCounter += 1
                                print("Download failed, retrying...", retryCounter)
                                time.sleep(1.0)
                            else:
                                keepDownloading = False
                                print("Download failed. Skipping image.")

    print("Total downloaded images:", totalDownloaded)

if removeDuplicates:

    # Stores the duplicates by hash:
    hashTable = {}

    # Check total files in directory:
    _, _, files = next(os.walk(outDirectory))

    # Natural sort the files:
    files = naturalSort(files, filenamePrefix)
    fileCount = len(files)

    print("Total Files in dir: " + str(fileCount))
    print("Checking valid images in: " + outDirectory)

    # Prepare the image hashing object:
    hsh = cv2.img_hash.BlockMeanHash_create()

    for filename in files:
        # Set the image path:
        imagePath = outDirectory + filename
        print("Reading image: ", imagePath)

        # Load the image:
        currentImage = cv2.imread(imagePath)

        if currentImage is None:
            if deleteFiles:
                print("Found invalid image, deleting...", filename)
                os.remove(imagePath)
        else:
            # Show the image:
            # showImage("Current Image", currentImage)

            # BGR to Gray:
            grayImage = cv2.cvtColor(currentImage, cv2.COLOR_BGR2GRAY)
            # Scale down the image by the scale factor:
            grayImage = cv2.resize(grayImage, None, fx=imageScale[0], fy=imageScale[0])

            # Compute image hash:
            imageHash = hsh.compute(currentImage)

            # Convert 32-bit array into string:
            hashString = getHashString(imageHash[0])
            # Get int from 32-bit hash:
            hashXOR = reduce(lambda x, y: x ^ y, imageHash[0])

            # Into the hash table:
            if hashString not in hashTable:
                hashTable[hashString] = [[currentImage, filename, hashXOR, True]]
            else:
                hashTable[hashString].append([currentImage, filename, hashXOR, False])

            # Show:
            # showImage("Current Image (Hashed)", currentImage)

    validImages = len(hashTable)
    validRate = (validImages / fileCount) * 100
    print("Valid images found: " + str(validImages) + " (" + f"{validRate:.4f}" + "%)")
    print("Removing Duplicates from: " + outDirectory)

    # Check out duplicates:
    deleteCounter = 0

    for key in hashTable:

        # Get dic entry:
        currentDuplicates = hashTable[key]
        # Check total duplicates:
        totalDuplicates = len(currentDuplicates)

        # For each image "group" (original + duplicates) use a unique number:
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))

        print("Examining duplicates for: " + str(key))
        duplicateCounter = 1

        for i in range(totalDuplicates):

            # Get the current duplicate:
            currentList = currentDuplicates[i]
            filename = currentList[1]
            hashXOR = currentList[2]

            # Counter String:
            imageText = str(duplicateCounter)

            # Keep only one of these duplicates, that is
            # the original image:
            removedText = ""

            if duplicateCounter > 1:
                # Actually delete the file:
                if deleteFiles:
                    # Set target path:
                    targetPath = outDirectory + filename
                    os.remove(targetPath)

                    # Debug mark:
                    removedText = "[X]"
                    deleteCounter += 1

            if showDuplicates:
                # Get the rest of the info:
                currentImage = currentList[0]

                # Set font parameters:
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 5
                fontThickness = 5

                # Print the info on the image:
                cv2.putText(currentImage, imageText + removedText, (5, 150), font, fontScale, color, fontThickness)
                # Show the image:
                showImage("Image Key: " + str(key), currentImage)

            # Show the info:
            print("Current:", key, "HashXOR", hashXOR, filename, "(" + imageText + ")", removedText)

            # Duplicate counter goes up:
            duplicateCounter += 1

    # Deleted all duplicates:
    duplicatesRate = (deleteCounter / fileCount) * 100
    print("Duplicates deleted: ", deleteCounter, "(" + f"{duplicatesRate:.4f}" + "%)")

    # Re-name files:
    renameCounter = 0
    imageCounter = 0
    newImagesCounter = 0

    for key in hashTable:

        print("Processing original image (First non-duplicate)")

        # Get dic entry:
        nonDuplicate = hashTable[key]
        currentList = nonDuplicate[0]
        currentImage = currentList[0]
        filename = currentList[1]

        # showImage("Original", currentImage)

        # Set new name:
        newName = filenamePrefix + str(imageCounter) + imageExtension

        # Full paths:
        oldPath = outDirectory + filename
        newPath = outDirectory + newName

        if oldPath != newPath:
            # Check the target is unique!
            alreadyExists = os.path.exists(newPath)
            while alreadyExists:
                imageCounter += 1
                newPath = outDirectory + filenamePrefix + str(imageCounter) + imageExtension
                alreadyExists = os.path.exists(newPath)

            # Rename:
            print("Renaming: ", oldPath, " to: ", newPath)
            os.rename(oldPath, newPath)
            renameCounter += 1

        # Ready for new file:
        imageCounter += 1
        newImagesCounter += 1

    renameRate = (renameCounter / fileCount) * 100
    print("Files renamed: ", renameCounter, "(" + f"{renameRate:.4f}" + "%)")
    fileTotal = fileCount - newImagesCounter
    print("Files Added (New): ", newImagesCounter)

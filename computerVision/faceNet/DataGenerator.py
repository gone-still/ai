# File        :   DataGenerator.py
# Version     :   0.0.1
# Description :   Data Generation for dataset streaming on the fly

# Date:       :   Oct 01, 2023
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   MIT

import numpy as np
import math
import tensorflow as tf
import cv2
from skimage.io import imread
from skimage.transform import resize
import time
from datetime import timedelta


class DataGenerator(tf.keras.utils.Sequence):
    @staticmethod
    # Shows an image
    def showImage(imageName, inputImage, delay=0):
        cv2.namedWindow(imageName, cv2.WINDOW_NORMAL)
        cv2.imshow(imageName, inputImage)
        cv2.waitKey(delay)

    # The init method creates the data generator and sets the generation
    # parameters:
    def __init__(self, dataset, labels, preprocessingFunction, preprocessingConfig, debug=False, batchSize=32,
                 imgSize=(100, 100),
                 shuffle=False, name="fucker"):
        # Initialization:
        self.dataset = dataset
        self.labels = labels
        self.batchSize = batchSize
        self.imgSize = imgSize
        # Preprocessing function:
        self.preprocessingFunction = preprocessingFunction
        # Preprocessing options dict:
        self.preprocessingConfig = preprocessingConfig
        # Debug option:
        self.debug = debug
        # Shuffles samples and labels on each new epoch:
        self.shuffle = shuffle
        # self.on_epoch_end()
        self.name = name

    # Computes the number of batches per epoch:
    def __len__(self):
        return math.ceil(len(self.dataset) / self.batchSize)

    # def on_epoch_begin(self):
    #     print(self.name)

    # Gets shuffled indices for this epoch:
    # def on_epoch_end(self):
    #     print(self.name)
        # self.indexes = np.arange(len(self.list_IDs))
        # if self.shuffle == True:
        #     np.random.shuffle(self.indexes)

    # Generates the current batch:
    def __getitem__(self, index):
        # print(self.name)
        # Compute slicing indices that will select the batch from
        # the dataset list:
        low = index * self.batchSize
        # Cap upper bound at array length; the last batch may be smaller
        # if the total number of items is not a multiple of batch size.
        high = min(low + self.batchSize, len(self.dataset))

        # print("Fuck:", self.name, low, high)

        # Slice the list of paths:
        currentDataset = self.dataset[low:high]
        # Slice the labels:
        currentLabels = self.labels[low:high]

        # Load and preprocess images:
        # Store batch samples here:
        batchSamples = []
        for currentPair in currentDataset:
            # Get current pair of paths:
            tempList = []
            for currentPath in currentPair:

                # Read image:
                # start = time.time()
                currentImage = cv2.imread(currentPath)
                # elapsed = (time.time() - start)
                # elapsed = str(timedelta(seconds=elapsed))
                # print("-> Loading time: ", elapsed)

                # print("readin image...")
                # Preprocess image:
                currentImage = self.preprocessingFunction(currentImage, self.imgSize,
                                                          self.preprocessingConfig["config"],
                                                          self.preprocessingConfig["auxFuns"])
                # currentImage = tf.keras.preprocessing.image.load_img(currentPath)
                # currentImage = tf.keras.preprocessing.image.img_to_array(currentImage)
                # currentImage = tf.keras.preprocessing.image.smart_resize(currentImage, size=self.imgSize)
                # elapsed = (time.time() - start)
                # elapsed = str(timedelta(seconds=elapsed))
                # print("-> Loading time: ", elapsed)
                tempList.append(currentImage)
            # Into the batch:
            batchSamples.append(tempList)

        # start = time.time()

        # Labels:
        batchLabels = np.array(currentLabels, dtype="float32")

        # Python list To numpy array of numpy arrays...
        batchSamplesArray = np.array(batchSamples)

        image1Arrays = batchSamplesArray[:, 0:1]
        image2Arrays = batchSamplesArray[:, 1:2]

        # Reshape the goddamn arrays: (drop "list dimension"):
        tempDim = image1Arrays.shape

        image1Arrays = image1Arrays.reshape(tempDim[0], tempDim[2], tempDim[3], tempDim[4])
        image2Arrays = image2Arrays.reshape(tempDim[0], tempDim[2], tempDim[3], tempDim[4])

        # Show the batch:
        if self.debug:
            for h in range(tempDim[0]):
                print("Sample: "+str(h), "Label: "+str(batchLabels[h]))
                self.showImage("[Batch] Sample 1", image1Arrays[h][0:self.imgSize[0]])
                self.showImage("[Batch] Sample 2", image2Arrays[h][0:self.imgSize[0]])

        # elapsed = (time.time() - start)
        # elapsed = str(timedelta(seconds=elapsed))
        # print("-> Batch time: ", elapsed)

        # Return the batch:
        return {"image1": image1Arrays, "image2": image2Arrays}, batchLabels

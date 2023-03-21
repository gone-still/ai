# Tensorflow CNN example for flower classification.
# Test file for model loading via OpenCV

# Import tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, GlobalAveragePooling2D, Activation
from tensorflow.keras.optimizers import Adam

# Import opencv & numpy
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Import keras2onnx and onnx
import onnx
import keras2onnx


# Function for displaying images of flowers from the data generators
def displayImages(data_generator, figTitle, no=15):
    sample_training_images, labels = next(data_generator)

    # Plot thee figure:
    plt.figure(figsize=[25, 25])

    # By default we're displaying 15 images, you can show more examples
    total_samples = sample_training_images[:no]

    cols = 5
    rows = int(np.floor(len(total_samples) / cols))

    for i, img in enumerate(total_samples, 1):
        plt.subplot(rows, cols, i)
        plt.imshow(img)

        # Converting One hot encoding labels to string labels and displaying it.
        class_name = classes[np.argmax(labels[i - 1])]
        plt.title(class_name)
        plt.axis('off')

    # show the plot:
    plt.show()


# Download the data set:
# url = ' https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
# filename = 'flower_photos.tgz'
# urllib.request.urlretrieve(url, filename)

# Load or create model from scratch:
loadModel = True
# Save ONNX model to disk:
saveOnnxModel = False

# Path to the data set:
dirPath = "flower_photos//"
# Path to the output files:
outputPath = "cnnTest//"

# Initialize classes list, this list will contain the names of our classes.
classes = []

totalImages = 0
# Iterate over the names of each class
for class_name in os.listdir(dirPath):

    # Get the full path of each class
    class_path = os.path.join(dirPath, class_name)

    # Check if the class is a directory/folder
    if os.path.isdir(class_path):
        # Get the number of images in each class and print them
        No_of_images = len(os.listdir(class_path))
        print("Found {} images of {}".format(No_of_images, class_name))
        totalImages = totalImages + No_of_images

        # Also store the name of each class
        classes.append(class_name)

# Sort the list in alphabatical order and print it
classes.sort()
print("Total Images: " + str(totalImages))
print(classes)

# Set the batch size, width, height and the percentage of the validation split,
# use 20% of training images as data augmentations:

batch_size = 60
IMG_HEIGHT = 224
IMG_WIDTH = 224
split = 0.2

#  Setup the ImagedataGenerator for training, pass in any supported augmentation schemes,
#  notice that we're also splitting the data with split argument.

datagen_train = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    validation_split=split,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='reflect')

# Setup the ImagedataGenerator for validation, no augmentation is done, only rescaling is done,
# notice that we're also splitting the data with split argument.

datagen_val = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    validation_split=split)

# Data Generation for Training with a constant seed valued 40,
# notice that we are specifying the subset as 'training'

# The "subset" variable tells the Imagedatagerator class which generator gets 80% and which gets 20% of the data

train_data_generator = datagen_train.flow_from_directory(batch_size=batch_size,
                                                         directory=dirPath,
                                                         shuffle=True,
                                                         seed=40,
                                                         subset='training',
                                                         interpolation='bicubic',
                                                         target_size=(IMG_HEIGHT, IMG_WIDTH))

# Data Generator for validation images with the same seed to make sure there is no data overlap,
# notice that we are specifying the subset as 'validation'
vald_data_generator = datagen_val.flow_from_directory(batch_size=batch_size,
                                                      directory=dirPath,
                                                      shuffle=True,
                                                      seed=40,
                                                      subset='validation',
                                                      interpolation='bicubic',
                                                      target_size=(IMG_HEIGHT, IMG_WIDTH))


# Model creation

# First Reset the generators, since we used the first batch to display the images.
vald_data_generator.reset()
train_data_generator.reset()

if loadModel:

    print("Loading Model...")

    model = load_model(outputPath + "flowers.h5")

    print("Model Loaded.")

else:

    print("Generating Model...")

    # Here we are creating Sequential model also defining its layers
    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(),
        Dropout(0.10),

        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),

        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),

        Conv2D(128, 3, padding='same', activation='relu'),
        MaxPooling2D(),

        Conv2D(256, 3, padding='same', activation='relu'),
        MaxPooling2D(),

        GlobalAveragePooling2D(),
        Dense(1024, activation='relu'),
        Dropout(0.10),
        Dense(len(classes), activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # See the summary:
    model.summary()

    print("Training Model...")

    # Model training:

    # Define the epoch number
    epochs = 60

    # Start Training
    history = model.fit(train_data_generator, steps_per_epoch=train_data_generator.samples // batch_size, epochs=epochs,
                        validation_data=vald_data_generator,
                        validation_steps=vald_data_generator.samples // batch_size)

    # Use model.fit_generator() if using TF version &lt; 2.2

    # Saving your model to disk allows you to use it later
    model.save(outputPath + "flowers.h5")

    # Plot the accuracy and loss curves for both training and validation

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training loss')
    plt.legend()

    plt.show()

# Model testing:

print("Testing Model...")

# Read the test image
img = cv2.imread("sunflower.jpg")

# Resize the image to the size you trained on.
imgr = cv2.resize(img, (224, 224))

# Convert image BGR TO RGB, since OpenCV works with BGR and tensorflow in RGB.
imgrgb = cv2.cvtColor(imgr, cv2.COLOR_BGR2RGB)

# Normalize the image to be in range 0-1 and then convert to a float array.
final_format = np.array([imgrgb]).astype('float64') / 255.0

# Perform the prediction
pred = model.predict(final_format)

# Get the index of top prediction
index = np.argmax(pred[0])

# Get the max probablity for that prediction
prob = np.max(pred[0])

# Get the name of the predicted class using the index
label = classes[index]

# Display the image and print the predicted class name with its confidence.
print("Predicted Flower is : {} {:.2f}%".format(label, prob * 100))
plt.imshow(img[:, :, ::-1])
plt.axis("off")
plt.show()

if saveOnnxModel:
    print("Saving ONNX Model...")

    # Convert model into onnx
    onnx_model = keras2onnx.convert_keras(model, model.name)

    # Save the model as flower.onnx
    onnx.save_model(onnx_model, outputPath + "flowers_model.onnx")

print("Loading ONNX Model...")

print("Testing Model via OpenCV...")

# Read the ONNX model
net = cv2.dnn.readNetFromONNX(outputPath + "flowers_model.onnx")

# Define class names and sort them alphabatically as this is how tf.keras remembers them
label_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
label_names.sort()

# Read the image
img_original = cv2.imread("sunflower.jpg")
img = img_original.copy()

# Resize Image
img = cv2.resize(img_original, (224, 224))

# Convert BGR TO RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Normalize the image and format it
img = np.array([img]).astype('float64') / 255.0

# Input Image to the network
net.setInput(img)

# Perform a Forward pass
Out = net.forward()

# Get the top predicted index
index = np.argmax(Out[0])

# Get the probability of the class.
prob = np.max(Out[0])

# Get the class name:
label = label_names[index]
text = "Predicted: {} {:.2f}%".format(label, prob)

# Let's check out the classification via OpenCV:
print("OpenCV>> Input belongs to class: " + str(label) + " (" + str(prob * 100) + " %)")

cv2.imshow("Classification Input", img_original)
cv2.waitKey()

import matplotlib

matplotlib.use("Agg")

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from cnn_classifier.lenet_classifier import LeNet  # my class for the model
from imutils import paths
import matplotlib.pyplot as plot
import numpy as np
import argparse
import random
import os
import cv2 as open_cv

arg_parse = argparse.ArgumentParser()
arg_parse.add_argument("-d", "--dataset", required=True, help="path to dataset")
arg_parse.add_argument("-m", "--model", required=True, help="path to output model")
arg_parse.add_argument("-p", "--plot", type=str, default="plot.png",
                       help="accuracy/loss plot")
args = vars(arg_parse.parse_args())

# initialize training
print("1.Initializing training....")
EPOCHS = 25
INIT_LEARN_RATE = 1e-3
BATCH_SIZE = 32

data = []
labels = []

print("2.Initializing images...")
image_paths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(image_paths)

for image_path in image_paths:
    image = open_cv.imread(image_path)

    # resize image to 28x28 (required for the models architecture LeNet
    image = open_cv.resize(image, (28, 28))
    image = img_to_array(image)
    data.append(image)
    #print(str(image))

    label = image_path.split(os.path.sep)[-2]
    label = 1 if label == "like" else 0
    labels.append(label)

# scale pixel densities to [0,1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# split data into training/testing data 75% to 25%
(trainData, testData, trainLabels, testLabels) = train_test_split(data, labels, test_size=0.25, random_state=42)

#convert labels to vectors
trainLabels = to_categorical(trainLabels, num_classes=2)
testLabels = to_categorical(testLabels, num_classes=2)

# create image generator for data augmentation
# this will change the images rotation, width height tilt etc to achieve better
# performance on a small dataset
augment = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                             horizontal_flip=True, fill_mode="nearest")

# initalizing the model
print("3.Initializing model...")
model = LeNet.init_model(img_width=28, img_height=28, channel_depth=3, classes=2)
optimizer = Adam(lr=INIT_LEARN_RATE, decay=INIT_LEARN_RATE / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

#train NN
print("4.Training model...")
H = model.fit_generator(augment.flow(trainData, trainLabels, batch_size=BATCH_SIZE),
                        validation_data=(testData, testLabels), steps_per_epoch=len(trainData) // BATCH_SIZE,
                        epochs=EPOCHS, verbose=1)

#save model
model.save(args["model"])
print("5.Model saved...")


print("Creating plot...")
plot.style.use("ggplot")
plot.figure()
N = EPOCHS
plot.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plot.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plot.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plot.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plot.title("Training Loss and Accuracy on Like/Not Like")
plot.xlabel("Epoch #")
plot.ylabel("Loss/Accuracy")
plot.legend(loc="lower left")
plot.savefig(args["plot"])

print("FINISHED")


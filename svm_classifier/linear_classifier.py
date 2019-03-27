from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2 as open_cv
import os
import pickle


class LinearSVCModel:

    @staticmethod
    def trainModel(trainImagesPaths, outputFileNameData = None, outputFileNameLabels = None):
        print("Getting histograms...")
        (data, labels) = getHistograms(trainImagesPaths)

        print("Saving model to file...")
        if outputFileNameData is not None:
            with open(outputFileNameData, "wb") as dataFile:
                pickle.dump(data, dataFile, pickle.HIGHEST_PROTOCOL)
        if outputFileNameLabels is not None:
            with open(outputFileNameLabels, "wb") as labelsFile:
                pickle.dump(labels, labelsFile, pickle.HIGHEST_PROTOCOL)


    @staticmethod
    def getModel(histogramFileName, labelFileName):
        (data,labels) = getLabeledData(histogramFileName, labelFileName)

        labelEncoder = LabelEncoder()
        labels = labelEncoder.fit_transform(labels)

        model = LinearSVC()
        model.fit(data, labels)

        return (model, labelEncoder)


def getLabeledData(dataFileName, labelFileName):
    labels = []
    data = []

    with open(dataFileName, "rb") as dataFile:
        data = pickle.load(dataFile)

    with open(labelFileName, "rb") as labelsFile:
        labels = pickle.load(labelsFile)

    return data, labels


def getHistograms(imgPaths):
    data = []
    labels = []

    total_images = len(imgPaths)
    count = 0

    # get color feature histogram from each image
    # get label for each image
    print("Extracting features of the images...")
    for imgPath in imgPaths:
        count += 1
        image = open_cv.imread(imgPath)
        label = imgPath.split(os.path.sep)[-2]

        histogram = extract_color_histogram(image)
        data.append(histogram)
        labels.append(label)

        if count % 20 == 0:
            print("Processed {:3d}/{:3d}".format(count, total_images))

    return (data, labels)


def extract_color_histogram(image, bins=(8, 8, 8)):
    """this method extracts the 3D color histogram from the HSV color space and returns it"""

    hsv = open_cv.cvtColor(image, open_cv.COLOR_BGR2HSV)
    histogram = open_cv.calcHist([hsv], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])

    if imutils.is_cv2():
        histogram = open_cv.normalize(histogram)
    else:
        open_cv.normalize(histogram, histogram)

    return histogram.flatten()  # this is the flatten color histogram of the image


# # init command line arguments
# argParser = argparse.ArgumentParser()
# argParser.add_argument("-d", "--dataset", required=True, help="path to dataset")
# args = vars(argParser.parse_args())
#
# print(args["dataset"])
# imgPaths = list(paths.list_images(args["dataset"]))
#
# data = []
# labels = []
#
# total_images = len(imgPaths)
# count = 0
#
# #get color feature histogram from each image
# #get label for each image
# print("Extracting features of the images...")
# for imgPath in imgPaths:
#     count += 1
#     image = open_cv.imread(imgPath)
#     label = imgPath.split(os.path.sep)[-2]
#
#     histogram = extract_color_histogram(image)
#     data.append(histogram)
#     labels.append(label)
#
#     if count%20 == 0:
#         print("Processed {:3d}/{:3d}".format(count, total_images))
#
#
# #convert labels from strings to integers
# labelEncoder = LabelEncoder()
# labels = labelEncoder.fit_transform(labels)
#
#
# #splitting data into training and testing
# (trainData, testData, trainLabels, testLabels) = \
#     train_test_split(np.array(data), labels, test_size=0.25, random_state=42)
#
#
# #train the classifier
# print("Training classifier...")
# model = LinearSVC()
# model.fit(trainData, trainLabels)
#
# #evaluate classifier
# print("Evaluating classifier...")
# predictions = model.predict(testData)
# print(classification_report(testLabels, predictions, target_names=labelEncoder.classes_))


import numpy as np
import argparse
from svm_classifier.linear_classifier import LinearSVCModel
from sklearn.metrics import classification_report
from imutils import paths
import svm_classifier.linear_classifier as linClassifier
from sklearn.preprocessing import LabelEncoder



argParser = argparse.ArgumentParser()
argParser.add_argument("-d", "--data", help="path to data")
argParser.add_argument("-l", "--labels", help="path to labels")
argParser.add_argument("-t", "--test", help="path to test data")
args = vars(argParser.parse_args())

histogramFileName = args["data"]
labelsFileName = args["labels"]
testDataPath = args["test"]

(model, labelEncoder) = LinearSVCModel.getModel(histogramFileName, labelsFileName)

(testData, testLabels) = linClassifier.getHistograms(list(paths.list_images(testDataPath)))
predictions = model.predict(testData)
labelEncoder = LabelEncoder()
testLabels = labelEncoder.fit_transform(testLabels)

print(classification_report(testLabels, predictions, target_names=labelEncoder.classes_))




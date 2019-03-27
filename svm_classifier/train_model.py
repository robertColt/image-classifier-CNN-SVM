from svm_classifier.linear_classifier import LinearSVCModel
from imutils import paths
import argparse


ARG_DATASET = "dataset"
ARG_DATASET_OUT = "datasetout"
ARG_LABELS_OUT = "labelsout"

#init command line arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-d", "--{}".format(ARG_DATASET), required=True, help="path to dataset")
argParser.add_argument("-o", "--{}".format(ARG_DATASET_OUT), required=True, help="path to data output file")
argParser.add_argument("-l", "--{}".format(ARG_LABELS_OUT), required=True, help="path to label output file")
args = vars(argParser.parse_args())


# print(args["dataset"])

imgPaths = list(paths.list_images(args[ARG_DATASET]))
dataOutPutFileName = args[ARG_DATASET_OUT]
labelsOutputFileName = args[ARG_LABELS_OUT]

# imgPaths = "../dataset/training"
# dataOutPutFileName = "cnn_classifier/model.data"
# labelsOutputFileName = "cnn_classifier/model.labels"


LinearSVCModel.trainModel(imgPaths, dataOutPutFileName, labelsOutputFileName)
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
from imutils import paths
import cv2
import os

argParser = argparse.ArgumentParser()
argParser.add_argument("-m", "--model", required=True,
                       help="path to trained model")
argParser.add_argument("-i", "--image", required=False,
                       help="path to test image")
argParser.add_argument("-d", "--dataset", required=True, help="path to test images")
args = vars(argParser.parse_args())

test_image_paths = sorted(list(paths.list_images(args["dataset"])))

print("1.Checking images...")
for imgPath in test_image_paths:
    delete = False
    try:
        img = cv2.imread(imgPath)
        delete = img is None
    except:
        # could not open image
        delete = True

    if delete:
        os.remove(imgPath)
        test_image_paths.remove(imgPath)

total_images = len(test_image_paths)
correct = 0
for imgPath in test_image_paths:
    # print("Predicting {} ...".format(imgPath))
    test_image = cv2.imread(imgPath)
    original_img = test_image.copy()

    # preprocessing img for classification
    test_image = cv2.resize(test_image, (28, 28))
    test_image = test_image.astype("float") / 255.0
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    # make prediction
    model = load_model(args["model"])
    (not_like, like) = model.predict(test_image)[0]

    label = "You Like This" if like > not_like else "You DON'T Like this"
    actual_label = imgPath.split("\\")[-2]
    if actual_label == "like" and label == "You Like This"\
            or actual_label=="not_like" and label == "You DON'T Like this":
        correct += 1


    probability = like if like > not_like else not_like
    label = "{}: {:.2f}%".format(label, probability * 100)

    print("{} - {}\n".format(imgPath, label))


print("Accuracy : {}/{} - {:.2f}%".format(correct, total_images, correct/total_images*100))

# output = imutils.resize(original_img, width=400)
# cv2.putText(output, label, (10,25), cv2.FONT_ARIAL, 0.7, (198, 215, 242), 2)
# cv2.imshow("Prediction", output)
# cv2.waitKey(0)

import requests
from imutils import paths
import argparse
import cv2
import os

argparser = argparse.ArgumentParser()
argparser.add_argument("-u", "--urls", required=True,
					   help="path to file containing image URLs")
argparser.add_argument("-o", "--output", required=True,
					   help="path to output directory of images")
args = vars(argparser.parse_args())


urls = open(args["urls"]).read().strip().split("\n")
no_img = 0
total_img = len(urls)

print("Downloading images...")
#download images
for url in urls:
    try:
        request = requests.get(url, timeout = 60)
        path = os.path.sep.join([args["output"], "{}.jpg".format(str(no_img).zfill(4))])
        file = open(path, "wb")
        file.write(request.content)
        file.close()

        no_img += 1
        print("{}/{}".format(no_img, total_img))
    except:
        print("Error occured on : " + url)


print("Checking images...")
#check if image can be opened with openCV
for image_path in paths.list_images(args["output"]):
    delete = False

    try:
        image = cv2.imread(image_path)

        if image is None:
            delete = True
    except:
        print("Cannot open image")
        delete = True

    if delete:
        os.remove(image_path)


print("FINISHED")
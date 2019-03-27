"""Colt Robert ICA gr.246

this is an image classifier build with the LeNet architecture for cnn_classifier's
I will try to train a model to learn what pictures a particular person likes
"""

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K_backend


class LeNet:
    """ LeNet model to classify images with CNNs"""

    @staticmethod
    def init_model(img_width, img_height, channel_depth, classes):
        """ initializes the model
        channel_depth: int
            1 if using greyscale, 3 for RGB
        classes: int
            number of classes to recognize"""
        model = Sequential() #using sequential class -> adding layers sequentially
        input_shape = (img_height, img_width, channel_depth)

        if K_backend.image_data_format() == "channels_first":
            input_shape = (channel_depth, img_height, img_width)

        #add CONV -> RELU -> POOL layers to the model

        #first layer
        conv_filter_size = (5,5)
        conv_filter_no = 20

        model.add(Conv2D(conv_filter_no, conv_filter_size, padding="same", input_shape = input_shape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        #second layer
        conv_filter_size = (5,5)
        conv_filter_no = 50
        model.add(Conv2D(conv_filter_no, conv_filter_size, padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        #flatten
        model.add(Flatten())  #flatten into fully connected ayer
        model.add(Dense(500)) #layers containing 500 nodes
        model.add(Activation("relu"))
        model.add(Dense(classes))  #another fully connected layer 1 node for each class
        model.add(Activation("softmax"))

        return model

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.mobilenet import MobileNet

from keras.layers import Flatten, Dense, Dropout, Input
from keras.models import Sequential, Model
from keras.models import load_model

import argparse
'''
# Map model names to classes
MODELS = {
    "vgg16": VGG16,
    "resnet": ResNet50,
    "mobilenet": MobileNet
}
'''
# Define path to pre-trained classification block weights - this is
vgg_weights_path = "weights/snapshot_vgg_weights.hdf5"
res_weights_path = "weights/snapshot_res_weights.hdf5"
mob_weights_path = "weights/snapshot_mob_weights.hdf5"
cnn_weights_path = "weights/snapshot_cnn_weights.hdf5"

def create_model(model_weights_path=cnn_weights_path, top_model=True, color_mode="rgb", input_shape=None):
    """Create custom model for transfer learning

    Steps:
    (i) load pre-trained NN architecture
    (ii) (optional) add custom classification block of two fully connected layers
    (iii) load pre-trained model weights, if available

    Parameters
    ----------
    model: str
        choose which pre-trained Keras deep learning model to use for the 'bottom' layers of the custom model
    model_weights_path: str
        optional path to weights for classification block; otherwise, pre-trained weights will be loaded
    top_model: bool
        whether to include custom classification block, or to load model 'without top' to extract features
    color_mode: str
        whether the image is gray scale or RGB; this will determine number of channels of model input layer

    Returns
    -------# train model

    my_model: keras.model
        Model utilised for prediction or training
    """

    # ensure a valid model name was supplied
    #if model not in MODELS.keys():
    #    raise AssertionError("The model parameter must be a key in the `MODELS` dictionary")

    # gray scale or color
    if color_mode == "grayscale":
        num_channels = 1
    else:
        num_channels = 3

    # Create pre-trained model for feature extraction, without classification block
    print("INFO: loading model")
    model_load=load_model.load_weights("Data/model.h5")
    print("INFO: Loaded model from disk")
    model = model_load(input_shape=(200,200,3))
    '''
    model = MODELS[model](include_top=True,
                          input_shape=(200, 200, 3))
    '''
    # For transfer learning
    if top_model:
        # Create classification block
        top_model = Sequential()
        top_model.add(Convolution2D(nb_filters,
                                (kernel_size[0], kernel_size[1]),
                                input_shape=input_shape))
        top_model.add(Activation('relu'))
        top_model.add(Convolution2D(nb_filters2, (kernel_size[0], kernel_size[1])))
        top_model.add(Activation('relu'))
        #model.add(MaxPooling2D(pool_size=pool_size))
        #model.add(Convolution2D(nb_filters3, (kernel_size[0], kernel_size[1])))
        #model.add(Activation('relu'))
        top_model.add(MaxPooling2D(pool_size=pool_size))
        top_model.add(Flatten(input_shape=model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.25))
        top_model.add(Dense(24, activation='softmax'))

        # Load weights for classification block
        print("INFO: loading model weights")
        '''
        if model_weights_path is not None:
            # user-supplied weights
            top_model.load_weights(model_weights_path)
        elif model == "vgg16":
            # pre-trained weights for transfer learning with VGG16
            top_model.load_weights(vgg_weights_path)
        elif model == "resnet":
            # pre-trained weights for transfer learning with ResNet50
            top_model.load_weights(res_weights_path)
        elif model == "mobnet":
            # pre-trained weights for transfer learning with ResNet50
            top_model.load_weights(mob_weights_path)
        '''

        # Join pre-loaded model + classification block
        print("INFO: creating model")
        my_model = Model(inputs=model.input,
                         outputs=top_model(model.output))
        return my_model
    else:
        return model

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.layers import Flatten, Dense, Dropout, Conv2D, Convolution2D, Activation, MaxPooling2D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, Callback
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adadelta, Adagrad, Adam, Adamax, Nadam
from keras import backend as K

import numpy as np
import argparse
import os

from matplotlib import pyplot
from numpy import array


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--train", required=False, default='../Data/TrainData',
                help="path to the train image directory")
ap.add_argument("-test", "--test", required=False, default='../Data/TestData',
                help="path to the test image directory")
ap.add_argument("-o", "--out_dir", default='../Data/OutputData',
                help="directory to output features")
ap.add_argument("-b", "--batch_size",
                type=int, default=16,
                help="batch size")
ap.add_argument("-e", "--epochs",
                type=int, default=1,
                help="number of epochs")
ap.add_argument("-c", "--channels",
                type=int, default=3,
                help="choose number of channels (1 or 3); gray scale or color images")
args = vars(ap.parse_args())

def mainloop():
    #################
    train_data_dir = args['train']
    validation_data_dir = args['test']
    batch_size = args["batch_size"]
    nb_classes = 24
    nb_epoch = args["epochs"]
    nb_train_samples = 7260
    nb_validation_samples = 240
    #image_size = 244
    img_width, img_height = 200, 200
    num_channels = args["channels"]
    if num_channels ==3:
        colour = "rgb"
    elif num_channels == 1:
        colour = "grayscale"

    #################

    # input image dimensions
    #img_rows, img_cols = image_size, image_size

    # number of convolutional filters to use
    nb_filters = 32
    nb_filters2 = 64
    nb_filters3 = 128

    # size of pooling area for max pooling
    pool_size = (2, 2)
    #convolution kernel size
    kernel_size = (3, 3)

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    #input_shape = (img_rows, img_cols, num_channels)

    model = Sequential()

    model.add(Convolution2D(nb_filters,
                            (kernel_size[0], kernel_size[1]),
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters2, (kernel_size[0], kernel_size[1])))
    model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=pool_size))
    #model.add(Convolution2D(nb_filters3, (kernel_size[0], kernel_size[1])))
    #model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adadelta(),
                  metrics=['accuracy'])
    '''
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    '''
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    print("model loaded.")

    # TRAINING

    # Model Checkpoint
    filepath = "back_up_" + "_weights.hdf5"

    save_snapshots = ModelCheckpoint(filepath,
                                     monitor='val_acc',
                                     save_best_only=True,
                                     save_weights_only=True,
                                     mode='max',
                                     verbose=0)


    # Save loss history
    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.losses = []
            self.accuracy = []

        def on_batch_end(self, batch, logs={}):
            self.losses.append(logs.get('loss'))
            self.accuracy.append(logs.get('acc'))

    loss_history = LossHistory()
    #callbacks_list = [loss_history]

    # define train data generator
    train_datagen = ImageDataGenerator(rescale=1.,
                                       featurewise_center=True,
                                       rotation_range=15.0,
                                       width_shift_range=0.15,
                                       height_shift_range=0.15)
    '''
    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    '''
    train_datagen.mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape(1, 1, 3)

    train_generator = train_datagen.flow_from_directory(
                args['train'],
                target_size=(img_width, img_height),
                batch_size=batch_size,
                class_mode="categorical",
                color_mode=colour,
                shuffle=False
                )
    '''
    train_generator = train_datagen.flow_from_directory(
        args['train'],
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')
    '''
    # define validation data generator
    test_datagen = ImageDataGenerator(rescale=1.,
                                      featurewise_center=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    #test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_datagen.mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape(1, 1, 3)

    test_generator = test_datagen.flow_from_directory(
                args['test'],
                target_size=(img_width, img_height),
                batch_size=batch_size,
                class_mode="categorical",
                color_mode=colour,
                shuffle=False
                )
    '''
    test_generator = test_datagen.flow_from_directory(
        args['test'],
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')
    '''
    ##############################
    steps_per_epoch = int(train_generator.samples//batch_size)
    validation_steps = int(test_generator.samples//batch_size)
    ##############################

    # train model
    history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=nb_epoch,
        verbose=1,
        #callbacks=callbacks_list,
        validation_data=test_generator,
        validation_steps=validation_steps,
        class_weight=None,
        pickle_safe=False)
    '''
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=nb_epoch,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size, shuffle=False)
    '''
    #save_loss_history = loss_history.losses
    #save_accuracy_history = loss_history.accuracy
    #np.savetxt("loss_history.txt", save_loss_history, delimiter=",")
    #np.savetxt("accuracy_history.txt", save_accuracy_history, delimiter=",")
    #model_json = model.to_json()
    #with open("Data/model.json", "w,") as json_file:
    #    json_file.write(simplejson.dumps(simmplejson.loads(model_json), indent=4))
    model.save_weights('snapshot_cnn_weights.hdf5')
    model.save('cnn_model.h5')
    print("Saved model to disk")

    evaluation_cost = history.history['val_loss']
    evaluation_accuracy = history.history['val_acc']
    training_cost = history.history['loss']
    training_accuracy = history.history['acc']
    pyplot.plot(training_cost)
    pyplot.plot(evaluation_cost)
    pyplot.title('model train vs validation loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.show()
    np.save("evaluation_cost.npy", evaluation_cost)
    np.save("evaluation_accuracy.npy", evaluation_accuracy)
    np.save("training_cost.npy", training_cost)
    np.save("training_accuracy.npy", training_accuracy)
    return None

if __name__ == "__main__":
    mainloop()

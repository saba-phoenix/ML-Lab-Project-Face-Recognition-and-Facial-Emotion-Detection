import io
import cv2
from PIL import Image, ImageFile
# import tensorflow as tf


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.platform import gfile

import numpy as np
import os
import pickle
import pandas as pd
import sys
import math
import multiprocessing as mp
import json
import requests
ImageFile.LOAD_TRUNCATED_IMAGES = True
cores = mp.cpu_count()

WORK_DIR = '/content/drive/My Drive/Projects/FR/1/'
MODEL_DIR = '/content/drive/My Drive/Courses/Machine Learning Lab/'


#****************************
#*******FACE DETECTION*******
#****************************
# TENSORFLOW FACE MODEL DETECTION MODEL FILE
PATH_TO_FACE_DETECTION_MODEL = MODEL_DIR + 'frozen_inference_graph_face.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'face_label_map.pbtxt'
# Haar cascade file path
CASCADE_FILE_PATH = MODEL_DIR + 'haarcascade_frontalface_default.xml'
# Face Detection Confidence
FACE_DETECTION_CONFIDENCE = 0.50
# Number of classes
NUM_CLASSES = 2
# FRACTION OF GPU MEMORY TO USE FOR DETECTION
FRACTION_GPU_MEMORY_DETECTION = 0.45
#******************************
#*******FACE RECOGNITION*******
#******************************
# PATH TO FACE ENCODINGS FILE
PATH_TO_FACE_ENCODINGS_FILE = MODEL_DIR + 'encodings.pickle'
# TENSORFLOW FACE MODEL RECOGNITION MODEL FILE
PATH_TO_FACE_RECOGNITION_MODEL = MODEL_DIR + '20170512-110547.pb'
# FRACTION OF GPU MEMORY TO USE FOR RECOGNITION
FRACTION_GPU_MEMORY_RECOGNITION = 0.45
# FACE RECOGNITION DISTANCE THRESHOLD
DISTANCE_THRESHOLD = 1
# Face recognition percentage threshold
PERCENTGE_THRESHOLD = 70
# Input Image Size
INPUT_IMAGE_SIZE = 160
# minimum face width and height
MIN_FACE_SIZE = 40
# BLURR THRESHOLD
BLURR_THRESHOLD = 100
# known person list for validation
KNOWN_PERSONS_VALIDATION = []



import sys
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, add, Dense, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, AveragePooling2D, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

sys.setrecursionlimit(2 ** 20)
np.random.seed(2 ** 10)



class WideResNet:
    def __init__(self, image_size, depth=16, k=8):
        self._depth = depth
        self._k = k
        self._dropout_probability = 0
        self._weight_decay = 0.0005
        self._use_bias = False
        self._weight_init = "he_normal"
        self._channel_axis = -1
        self._input_shape = (image_size, image_size, 3)

    # Wide residual network http://arxiv.org/abs/1605.07146
    def _wide_basic(self, n_input_plane, n_output_plane, stride):
        def f(net):
            
            conv_params = [[3, 3, stride, "same"],
                           [3, 3, (1, 1), "same"]]

            n_bottleneck_plane = n_output_plane

            # Residual block
            for i, v in enumerate(conv_params):
                if i == 0:
                    if n_input_plane != n_output_plane:
                        net = BatchNormalization(axis=self._channel_axis)(net)
                        net = Activation("relu")(net)
                        convs = net
                    else:
                        convs = BatchNormalization(axis=self._channel_axis)(net)
                        convs = Activation("relu")(convs)

                    convs = Conv2D(n_bottleneck_plane, kernel_size=(v[0], v[1]),
                                          strides=v[2],
                                          padding=v[3],
                                          kernel_initializer=self._weight_init,
                                          kernel_regularizer=l2(self._weight_decay),
                                          use_bias=self._use_bias)(convs)
                else:
                    convs = BatchNormalization(axis=self._channel_axis)(convs)
                    convs = Activation("relu")(convs)
                    if self._dropout_probability > 0:
                        convs = Dropout(self._dropout_probability)(convs)
                    convs = Conv2D(n_bottleneck_plane, kernel_size=(v[0], v[1]),
                                          strides=v[2],
                                          padding=v[3],
                                          kernel_initializer=self._weight_init,
                                          kernel_regularizer=l2(self._weight_decay),
                                          use_bias=self._use_bias)(convs)

            # Shortcut Connection: identity function or 1x1 convolutional
            #  (depends on difference between input & output shape - this
            #   corresponds to whether we are using the first block in each
            #   group; see _layer() ).
            if n_input_plane != n_output_plane:
                shortcut = Conv2D(n_output_plane, kernel_size=(1, 1),
                                         strides=stride,
                                         padding="same",
                                         kernel_initializer=self._weight_init,
                                         kernel_regularizer=l2(self._weight_decay),
                                         use_bias=self._use_bias)(net)
            else:
                shortcut = net

            return add([convs, shortcut])

        return f


    # "Stacking Residual Units on the same stage"
    def _layer(self, block, n_input_plane, n_output_plane, count, stride):
        def f(net):
            net = block(n_input_plane, n_output_plane, stride)(net)
            for i in range(2, int(count + 1)):
                net = block(n_output_plane, n_output_plane, stride=(1, 1))(net)
            return net

        return f

#    def create_model(self):
    def __call__(self):

        assert ((self._depth - 4) % 6 == 0)
        n = (self._depth - 4) / 6

        inputs = Input(shape=self._input_shape)

        n_stages = [16, 16 * self._k, 32 * self._k, 64 * self._k]

        conv1 = Conv2D(filters=n_stages[0], kernel_size=(3, 3),
                              strides=(1, 1),
                              padding="same",
                              kernel_initializer=self._weight_init,
                              kernel_regularizer=l2(self._weight_decay),
                              use_bias=self._use_bias)(inputs)  # "One conv at the beginning (spatial size: 32x32)"

        # Add wide residual blocks
        block_fn = self._wide_basic
        conv2 = self._layer(block_fn, n_input_plane=n_stages[0], n_output_plane=n_stages[1], count=n, stride=(1, 1))(conv1)
        conv3 = self._layer(block_fn, n_input_plane=n_stages[1], n_output_plane=n_stages[2], count=n, stride=(2, 2))(conv2)
        conv4 = self._layer(block_fn, n_input_plane=n_stages[2], n_output_plane=n_stages[3], count=n, stride=(2, 2))(conv3)
        batch_norm = BatchNormalization(axis=self._channel_axis)(conv4)
        relu = Activation("relu")(batch_norm)

        # Classifier block
        pool = AveragePooling2D(pool_size=(8, 8), strides=(1, 1), padding="same")(relu)
        flatten = Flatten()(pool)
        predictions_g = Dense(units=2, kernel_initializer=self._weight_init, use_bias=self._use_bias,
                              kernel_regularizer=l2(self._weight_decay), activation="softmax")(flatten)
        predictions_a = Dense(units=101, kernel_initializer=self._weight_init, use_bias=self._use_bias,
                              kernel_regularizer=l2(self._weight_decay), activation="softmax")(flatten)

        model = Model(inputs=inputs, outputs=[predictions_g, predictions_a])

        return model
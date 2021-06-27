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


WORK_DIR = './'
MODEL_DIR = './'
if os.path.exists(MODEL_DIR + 'frozen_inference_graph_face.pb'):
    PATH_TO_FACE_DETECTION_MODEL = MODEL_DIR + 'frozen_inference_graph_face.pb'
if os.path.exists(MODEL_DIR + 'haarcascade_frontalface_default.xml'):
    CASCADE_FILE_PATH = MODEL_DIR + 'haarcascade_frontalface_default.xml'
if os.path.exists(MODEL_DIR + 'face_label_map.pbtxt'):
    PATH_TO_LABELS = 'face_label_map.pbtxt'
if os.path.exists(MODEL_DIR + '20170512-110547.pb'):
    PATH_TO_FACE_RECOGNITION_MODEL = MODEL_DIR + '20170512-110547.pb'




import cv2
import os
import numpy as np
from tensorflow.keras.utils import get_file
from tensorflow.keras.models import load_model
import copy

from WideResNet import WideResNet
from TensorflowFaceRecognition import TensorflowFaceRecognition


def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

def get_labels(dataset_name):
    if dataset_name == 'fer2013':
        return {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'Happy',
                4: 'sad', 5: 'surprise', 6: 'neutral'}
    elif dataset_name == 'imdb':
        return {0: 'woman', 1: 'man'}
    elif dataset_name == 'KDEF':
        return {0: 'AN', 1: 'DI', 2: 'AF', 3: 'HA', 4: 'SA', 5: 'SU', 6: 'NE'}
    else:
        raise Exception('Invalid dataset name')


class FaceImage(object):
    WRN_WEIGHTS_PATH = "https://github.com/Tony607/Keras_age_gender/releases/download/V1.0/weights.18-4.06.hdf5"
    EMOTION_MODEL_PATH = 'fer2013_mini_XCEPTION.119-0.65.hdf5'

    def __new__(cls, weight_file=None, depth=16, width=8, face_size=64):
        if not hasattr(cls, 'instance'):
            cls.instance = super(FaceImage, cls).__new__(cls)
        return cls.instance

    def __init__(self, depth=16, width=8, face_size=64):
        self.face_size = face_size
        print("Loading age and gender detection model...")
        from WideResNet import WideResNet
        self.model = WideResNet(face_size, depth=depth, k=width)()
        model_dir = os.path.join(os.getcwd(), "models").replace("//", "\\")
        fpath = get_file('weights.18-4.06.hdf5',
                         self.WRN_WEIGHTS_PATH,
                         cache_subdir=model_dir)
        self.model.load_weights(fpath)
        print("Loaded models")

        print('Loading emotion model...')
        os.system('wget https://github.com/oarriaga/face_classification/raw/master/trained_models/fer2013_mini_XCEPTION.119-0.65.hdf5')
        self.emotion_labels = get_labels('fer2013')
        self.emotion_classifier = load_model(self.EMOTION_MODEL_PATH, compile=False)
        self.emotion_target_size = self.emotion_classifier.input_shape[1:3]
        # self.emotion_offsets = (20, 40)
        print('Emotion model loaded')

    @classmethod
    def draw_label_bottom(cls, image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale=0.5, thickness=1, row_index=0, alpha=0.5):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        point = (point[0], point[1] + (row_index * size[1]))
        x, y = point
        overlay = image.copy()
        cv2.rectangle(overlay, (x, y), (x + size[0], y + size[1]), (255, 0, 0), cv2.FILLED)
        point = x, y+size[1]
        cv2.putText(overlay, label, point, font, font_scale, (255, 255, 255), thickness)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        
    def get_regular_face(self, img, bb):
        return img[bb[1]:bb[3], bb[0]:bb[2], :]

    def get_expanded_face(self, img, bb):
        img_h, img_w, _ = np.shape(img)
        x1, y1, x2, y2 = bb
        w, h = x2 - x1, y2 - y1
        xw1 = max(int(x1 - 0.4 * w), 0)
        yw1 = max(int(y1 - 0.4 * h), 0)
        xw2 = min(int(x2 + 0.4 * w), img_w - 1)
        yw2 = min(int(y2 + 0.4 * h), img_h - 1)
        return cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (self.face_size, self.face_size))

    def detect_face_char(self, img):
        # workaround for CV2 bug
        img = copy.deepcopy(img)
        
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_h, img_w, _ = np.shape(input_img)
        tRecognition = TensorflowFaceRecognition(PATH_TO_FACE_RECOGNITION_MODEL)
        # for face detection
        face_bbs = tRecognition.detect_face(input_img)

        expanded_face_imgs = np.empty((len(face_bbs), self.face_size, self.face_size, 3))
        emotion2_results = []
  
        # Get face images      
        for i, bb in enumerate(face_bbs):
            x1, y1, x2, y2 = bb
            w, h = x2 - x1, y2 - y1
            expanded_face_imgs[i, :, :, :] = self.get_expanded_face(img, bb)
            reg_face = self.get_regular_face(img, bb)
            gray_face = gray_image[y1:y2, x1:x2]
            
            gray_face = cv2.resize(gray_face, (self.emotion_target_size))

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_prediction = self.emotion_classifier.predict(gray_face)
            emotion_probability = np.max(emotion_prediction)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = self.emotion_labels[emotion_label_arg]
            emotion2_results.append(emotion_text)

        
        if len(expanded_face_imgs) > 0:
            # predict ages and genders of the detected faces
            results = self.model.predict(expanded_face_imgs)
            predicted_genders = results[0]
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results[1].dot(ages).flatten()
            
        # draw results
        for i, bb in enumerate(face_bbs):
        
            ## Display age, gender and emotion 
#             label2 = "{}, {}, {}".format(int(predicted_ages[i]),
#                                          "F" if predicted_genders[i][0] > 0.5 else "M",
#                                          emotion2_results[i])
              label2 = "{}".format(int(emotion2_results[i])
            self.draw_label_bottom(img, (bb[0], bb[1]), label2, row_index=0)

        # draw face rectangles
        for i, bb in enumerate(face_bbs):
            x1, y1, x2, y2 = bb
            w, h = x2 - x1, y2 - y1
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        return img
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

from TensorflowFaceRecognition import TensorflowFaceRecognition



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



class IdData(object):
    """Keeps track of known identities and calculates id matches"""

    def __init__(self):
        # Initializing the parameters
        self.distance_threshold = DISTANCE_THRESHOLD
        self.names = []
        self.known_encodings = []

    def fetchKnownEncodings(self, path):
        """getting the known encoding from a pickle file"""
        try:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                with open(path, 'rb') as fr:
                    [self.names, self.known_encodings] = pickle.load(fr)
        except Exception as e:
            print('Exception occured in reading from Pickle file:' + e)
    
    def saveEncoding(self, path, name, image):
        """saving known encoding to a pickle file"""
        try:
            data = tRecognition.getFace(image)
            print('Encodings successfully calculated!')
            if len(data) > 1:
                raise Exception('More than one person in the image found')
            if os.path.exists(path):
                with open(path, 'rb') as fr:
                    [self.names, self.known_encodings] = pickle.load(fr)
                    self.names.append(name)
                    self.known_encodings.append(data[0]['embeddings'][0])
                    with open(path, 'wb') as fw:
                        pickle.dump([self.names, self.known_encodings], fw, protocol=pickle.HIGHEST_PROTOCOL)
                    print('File saved')
            else:
                self.names.append(name)
                self.known_encodings.append(data[0]['embeddings'][0])
                with open(path, 'wb') as fw:
                    pickle.dump([self.names, self.known_encodings], fw, protocol=pickle.HIGHEST_PROTOCOL)
                print('File saved')
        except Exception as e:
            print('Exception occured in reading from Pickle file:' + e)
    
    def findPeople(self, features_arr, thres=0.5, percent_thres=PERCENTGE_THRESHOLD):
        """
        :param features_arr: a list of 128d Features of all faces on screen
        :param thres: distance threshold
        :param percent_thres: confidence percentage
        :return: person name and percentage
        """
        data_set = id_data.known_encodings
        known_persons = id_data.names
        returnRes = []
        for data in features_arr:
            result = "Unknown"
            smallest = sys.maxsize
            for (i, person_data) in enumerate(data_set):
                distance = np.sqrt(np.sum(np.square(person_data - data['embeddings'][0])))
                if distance < smallest:
                    smallest = distance
                    result = known_persons[i]
            percentage = min(100, 100 * thres / smallest)
            if percentage <= percent_thres:
                result = "Unknown"
            returnRes.append((result, percentage))
        return returnRes
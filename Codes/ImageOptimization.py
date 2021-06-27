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

class ImageOptimization(object):
    def __init__(self):
        super(ImageOptimization, self).__init__()

    # for illumination correction use this function on cropped face image
    def apply_clahe(self, frame):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        lab_img = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        lab = cv2.split(lab_img)
        lab[0] = clahe.apply(lab[0])
        _lab_img = cv2.merge(lab)
        rgb_img = cv2.cvtColor(_lab_img, cv2.COLOR_Lab2RGB)
        bgr_img = cv2.cvtColor(_lab_img, cv2.COLOR_Lab2BGR)
        # return bgr image
        return bgr_img

    # use this function to apply clache to cropped
    def apply_clahe_face(self, img, mtcnn_face):
        # mtcnn_face = mtcnn_faces[0] # x1,y1,x2,y2
        x1, y1, x2, y2 = mtcnn_face
        h, w = img.shape[:2]
        h_face = y2 - y1
        w_face = x2 - x1
        del_w = int(0.1 * float(w_face))
        del_h = int(0.1 * float(h_face))
        _x1 = max(0, x1 - del_w)
        _x2 = min(w - 1, x2 + del_w)
        _y1 = max(0, y1 - del_h)
        _y2 = min(h - 1, y2 + del_h)
        face = img[_y1:_y2, _x1:_x2]
        clahe_face = self.apply_clahe(face)
        clahe_img = img.copy()
        clahe_img[_y1:_y2, _x1:_x2] = clahe_face
        return clahe_img

    # this function applies 2D rotation
    def align_face_eyes(self, img, eyes):
        # angle between eyes
        dY = eyes[1][1] - eyes[0][1]
        dX = eyes[1][0] - eyes[0][0]
        angle = -np.degrees(np.arctan2(dY, dX))
        face_aligned_img = rotate_bound(img.copy(), angle)
        return face_aligned_img

    # for image brightness adjustment
    def adjust_gamma(self, img, gamma=1.0):
        """
        :param img  : give an input image
        :param gamma: desired gamma value
        :return LUt : look-up-table for pixel wise gamma correction
        """
        inv_gamma = 1.0 / gamma
        # building a lookup table
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 255)]).astype('uint8')
        # applying gamma correction using lookup table
        return cv2.LUT(img, table)
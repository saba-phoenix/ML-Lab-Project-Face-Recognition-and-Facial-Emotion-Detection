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

class TensoflowFaceDector(object):
    def __init__(self, PATH_TO_CKPT):
        """Tensorflow detector
        """
        print('Face Detection model loading...')
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            model_exp = os.path.expanduser(PATH_TO_CKPT)
            # print('Model filename: %s' % model_exp)
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        with self.detection_graph.as_default():
            # fraction of GPU memory to use
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FRACTION_GPU_MEMORY_DETECTION)
            # gpu_config = tf.ConfigProto(gpu_options=gpu_options)
            gpu_config = tf.ConfigProto()
            
            gpu_config.intra_op_parallelism_threads = cores  # no of physical cpu cores to allocate
            gpu_config.intra_op_parallelism_threads = cores  # no of physical cpu cores to allocate
            gpu_config.gpu_options.allow_growth = True

            self.sess = tf.Session(graph=self.detection_graph, config=gpu_config)
            # writer = tf.summary.FileWriter('./tf_logs/face_detection', self.sess.graph)
            self.windowNotSet = True
            print('Face Detection model loaded!')

    def run(self, image):
        """image: bgr image
        return (boxes, scores, classes, num_detections)
        """
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        return boxes, scores, classes, num_detections


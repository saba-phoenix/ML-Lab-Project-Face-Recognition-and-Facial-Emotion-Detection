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
from TensoflowFaceDector import TensoflowFaceDector

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





class TensorflowFaceRecognition(object):

    def __init__(self, MODEL_PATH):
        print('Face Recognition model loading...')
        self.recognition_graph = tf.Graph()
        with self.recognition_graph.as_default():
            model_exp = os.path.expanduser(MODEL_PATH)
            if os.path.isfile(model_exp):
                # print('Model filename: %s' % model_exp)
                with gfile.FastGFile(model_exp, 'rb') as f:
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(f.read())
                    tf.import_graph_def(graph_def, input_map=None, name='')
                    # Get input and output tensors
                    self.images_placeholder = self.recognition_graph.get_tensor_by_name("input:0")
                    self.embeddings = self.recognition_graph.get_tensor_by_name("embeddings:0")
                    self.phase_train_placeholder = self.recognition_graph.get_tensor_by_name("phase_train:0")
                    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.FRACTION_GPU_MEMORY_RECOGNITION)
                    # # creating the tensorflow session
                    # self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
                    self.sess = tf.Session()
                    print('Face Recognition model loaded!')
                    # writer = tf.summary.FileWriter('./tf_logs/face_recognition', self.sess.graph)
            else:
                print('Model directory: %s' % model_exp)
                meta_file, ckpt_file = get_model_filenames(model_exp)

                print('Metagraph file: %s' % meta_file)
                print('Checkpoint file: %s' % ckpt_file)

                saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file), input_map=input_map)
                saver.restore(self.recognition_graph.get_default_session(), os.path.join(model_exp, ckpt_file))

    def getEmbedding(self, resized, input_image_size=INPUT_IMAGE_SIZE):
        """Calculating the embedding for a face patch"""
        reshaped = resized.reshape(-1, input_image_size, input_image_size, 3)
        feed_dict = {self.images_placeholder: reshaped, self.phase_train_placeholder: False}
        embedding = self.sess.run(self.embeddings, feed_dict=feed_dict)
        return embedding
    
    def findCosineSimilarity(self, vector_1, vector_2):
        a = np.matmul(np.transpose(vector_1), vector_2)
        b = np.sum(np.multiply(vector_1, vector_2))
        c = np.sum(np.multiply(vector_1, vector_2))
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
    
    def l2_normalize(self, x):
        return x / np.sqrt(np.sum(np.multiply(x, x)))
    
    def prewhiten(self, x):
        """Normalzing the face patch"""
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1 / std_adj)
        return y

    def detect_face(self, img, thres=FACE_DETECTION_CONFIDENCE, input_image_size=INPUT_IMAGE_SIZE):
        h, w = img.shape[:2]
        bbs = []
        tDetector = TensoflowFaceDector(PATH_TO_FACE_DETECTION_MODEL)
        (boxes, scores, classes, num_detections) = tDetector.run(img)
        boxes = np.squeeze(boxes)
        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores)
        for i in range(boxes.shape[0]):
            if scores is None or scores[i] > thres:
                box = tuple(boxes[i].tolist())
                ymin, xmin, ymax, xmax = box
                left, right, top, bottom = xmin * w, xmax * w, ymin * h, ymax * h
                left, right, top, bottom = int(left), int(right), int(top), int(bottom)
                if right - left >= MIN_FACE_SIZE and bottom - top >= MIN_FACE_SIZE:
                    bbs.append([int(left), int(top), int(right), int(bottom)])
        return bbs

    def getFace(self, img, thres=FACE_DETECTION_CONFIDENCE, input_image_size=INPUT_IMAGE_SIZE):
        faces = []
        bbs = self.detect_face(img)
        for left, top, right, bottom in bbs:
            # cropping the face patch from the frame
            cropped = img[top:bottom, left:right, :]
            # resizing the cropped image to the input_image_size
            resized = cv2.resize(cropped, (input_image_size, input_image_size), interpolation=cv2.INTER_CUBIC)
            # normalizing the image
            prewhitened = self.prewhiten(resized)
            # making the output
            faces.append(
                {'face': resized, 'rect': (left, top, right, bottom),
                    'embeddings': self.getEmbedding(prewhitened)})
        return faces
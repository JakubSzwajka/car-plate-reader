import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# required library
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from os.path import splitext,basename
from keras.models import model_from_json
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelEncoder
import glob

# extra scripts
import settings
from local_utils import detect_lp
import plate_finder_methods as pfm
import letters_finder as lf
import letter_recognition as lr
import plotter

# below line for colab google only! 
# from importlib import reload  
# reload(glob)

# FIND PLATE
wpod_net = pfm.load_model_plate_recognition(settings.WPOD_NET_PATH)
vehicle, LpImg,cor = pfm.get_plate(settings.TEST_IMAGE_PATH, wpod_net)

if len(LpImg) > 0:
    print("Found " , len(LpImg) , " plates")
    plotter.plot_plate(vehicle, LpImg, cor)

    # FIND LETTERS 
    plate_image, gray, blur, binary, thre_mor = lf.grey_scale_img(LpImg)
    plotter.plot_grey_scale_plates(plate_image, gray, blur, binary, thre_mor)

    # # creat a copy version "test_roi" of plat_image to draw bounding box
    test_roi = plate_image.copy()
    crop_characters = plotter.get_letter_contours( binary , plate_image ,test_roi, thre_mor)

    # Load model architecture, weight and labels
    letter_model, labels = lr.load_models(settings.MODELS_PATH)
    print(lr.get_plate_string( crop_characters, letter_model, labels))

else: 
    print("No plates found!")
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
from local_utils import detect_lp, logger
import plate_finder_methods as pfm
import letters_finder as lf
import letter_recognition as lr
import plotter

# below line for colab google only! 
# from importlib import reload  
# reload(glob)

plates = []

# ----------- FIND PLATE ----------------------------------------------
# ----------- Load model for plate in image recognition ---------------
wpod_net = pfm.load_model_plate_recognition(settings.WPOD_NET_PATH)

# ----------- Load model architecture, weight and labels --------------
letter_model, labels = lr.load_models(settings.MODELS_PATH)

# ----------- Get Car/ LicencePlate/ and coordinates ------------------
LpImg = pfm.get_plate(settings.TEST_IMAGE_PATH, wpod_net)
logger("Found " + str(len(LpImg)) + " plates")

# ----------- for plate in plates array ------------------------------- 
for plate in LpImg:
    if settings.PLOT_IMGS: plotter.plot_plate(plate)
    # ------------- FIND LETTERS -------------------------------------- 
    plate_image, gray, blur, binary, thre_mor = lf.grey_scale_img(plate)
    if settings.PLOT_IMGS: plotter.plot_grey_scale_plates(plate_image, gray, blur, binary, thre_mor)

    # -------------- creat a copy version "test_roi" of plat_image to draw bounding box
    test_roi = plate_image.copy()
    crop_characters = lf.get_letter_contours( binary , plate_image ,test_roi, thre_mor)
    if settings.PLOT_IMGS: plotter.plot_letters_on_plate(test_roi, crop_characters)
    if settings.PLOT_IMGS: plotter.plot_binary_plate(crop_characters)

    plate = lr.get_plate_string( crop_characters, letter_model, labels)
    plates.append(plate)

logger(plates)
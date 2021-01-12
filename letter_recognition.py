from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
import cv2
import numpy as np
from local_utils import logger

def load_models( folder_path = ''):
    json_file = open(folder_path + 'MobileNets_character_recognition.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(folder_path + "License_character_recognition_weight.h5")
    logger("Letter recognition model loaded successfully...")

    labels = LabelEncoder()
    labels.classes_ = np.load(folder_path + 'license_character_classes.npy')
    logger("Letter recognition labels loaded successfully...")

    return model, labels

def predict_from_model(image,model,labels):
    image = cv2.resize(image,(80,80))
    image = np.stack((image,)*3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
    return prediction

def get_plate_string( crop_characters, letter_model, labels):
    final_string = ''
    for i,character in enumerate(crop_characters):
        title = np.array2string(predict_from_model(character,letter_model,labels))
        final_string+=title.strip("'[]")

    return final_string
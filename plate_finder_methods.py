import cv2
from local_utils import detect_lp, logger 
import os
from keras.models import model_from_json

# load plate recognition model
def load_model_plate_recognition(path):
    try:
        path = os.path.splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        logger("Loading model successfully...")
        return model
    except Exception as e:
        logger(e)

def preprocess_images(image_path,resize=False):
    images = []
    for filename in os.listdir(image_path):
        img = cv2.imread(os.path.join(image_path,filename))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255
            if resize: img = cv2.resize(img, (224,224))
            images.append(img)
    return images

def get_plate(image_path, model, Dmax=608, Dmin = 608):
    images = preprocess_images(image_path)
    plates = []
    
    for img in images:
        ratio = float(max(img.shape[:2])) / min(img.shape[:2])
        side = int(ratio * Dmin)
        bound_dim = min(side, Dmax)
        _ , LpImg, *_ = detect_lp(model, img, bound_dim, lp_threshold=0.5)
        plates += LpImg
    return plates

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from keras.models import Sequential
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.utils import img_to_array, load_img
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
import pickle as pk
from ultralytics import YOLO
import cv2
import json
import keras
import warnings
import gdown
import os
warnings.filterwarnings('ignore')

CLASS_INDEX = None
CLASS_INDEX_PATH = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'

model_urls = ["https://drive.google.com/uc?id=1xKyczvMpIVCx4XB72lH1sQlnvN_6n_Q8", "https://drive.google.com/uc?id=14fE5lDjroLDppP3EuPyipE3TlYumCGr0", "https://drive.google.com/uc?id=1UGyuOwtaaUtrXJPMfKXkZUqXa2tzxVF7", "https://drive.google.com/uc?id=17DvZPqdMaku7a_wrw2l7eq9qqTVlxTmG", "https://drive.google.com/uc?id=1VJe4GlOKjf9yj-QMuHyV7D5oFZCqLqGA"]
model_file_paths = ["Vehicle_Damaged_Decision_Model.h5", "Vehicle_Damage_Localization_Model.h5", "Vehicle_Damage_Severity_Model.h5", "Vehicle_Damage_Part_YOLOv8_FineTuned_Model.pt", "Vehicle_Damage_Type_YOLOv8_FineTuned_Model.pt"]

def download_models(model_url, model_file_path):
    if not os.path.exists(model_file_path):
        gdown.download(model_url, model_file_path, quiet=False)
    else:
        print(f"{model_file_path} already exists.")

for i in range(len(model_urls)):
    download_models(model_urls[i], model_file_paths[i])

model1 = VGG16(weights='imagenet')
model2 = Sequential()
model2.add(Conv2D(16, (3, 3), activation='relu', input_shape=(256,256, 3)))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Conv2D(32, (3, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Conv2D(64, (3, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Conv2D(128, (3, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Conv2D(256, (3, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Flatten())
model2.add(Dense(256, activation='relu'))
model2.add(Dropout(0.2))
model2.add(Dense(128, activation='relu'))
model2.add(Dropout(0.2))
model2.add(Dense(2, activation='softmax'))

model2.load_weights("./Vehicle_Damaged_Decision_Model.h5")
model3 = load_model("./Vehicle_Damage_Localization_Model.h5")
model4 = load_model("./Vehicle_Damage_Severity_Model.h5")
model_obj = YOLO("./Vehicle_Damage_Part_YOLOv8_FineTuned_Model.pt")
model_seg = YOLO("./Vehicle_Damage_Type_YOLOv8_FineTuned_Model.pt")

with open("./VGG16_Category_List.pk", 'rb') as f:
    cat_list = pk.load(f)

def prepare_image_224(img_path):
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return preprocess_input(x)

def prepare_img_256(img_path):
    img = load_img(img_path, target_size=(256, 256))
    x = img_to_array(img) / 255.0
    return np.expand_dims(x, axis=0)

def get_predictions(preds, top=5):
    global CLASS_INDEX
    if len(preds.shape) != 2 or preds.shape[1] != 1000:
        raise ValueError('`decode_predictions` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 1000)). '
                         'Found array with shape: ' + str(preds.shape))
    if CLASS_INDEX is None:
        fpath = keras.utils.data_utils.get_file('imagenet_class_index.json', CLASS_INDEX_PATH, cache_subdir='models')
        CLASS_INDEX = json.load(open(fpath))
    arr = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        indexes = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        indexes.sort(key=lambda x: x[2], reverse=True)
        arr.append(indexes)
    return arr

def pipe1(img_224, model):
    out = model.predict(img_224)
    preds = get_predictions(out, top=5)
    for pred in preds[0]:
        if pred[0:2] in cat_list:
            return True
        return False

def pipe2(img_256, model):
    preds = model.predict(img_256)
    return preds[0][0] >= 0.45

def pipe3_loc(img_256, model):
    pred = model.predict(img_256)
    labels = {0: 'front', 1: 'rear', 2: 'side'}
    return labels[np.argmax(pred)]

def pipe3_sev(img_256, model):
    pred = model.predict(img_256)
    labels = {0: 'minor', 1: 'moderate', 2: 'severe'}
    return labels[np.argmax(pred)]

def object_detect(img_path):
    result = model_obj.predict(img_path, conf=0.3)
    if result:
        result_image = cv2.cvtColor(result[0].plot(), cv2.COLOR_BGR2RGB)
        return "Object detection completed.", result_image
    return "No objects detected.", None

def segment(img_path):
    result = model_seg.predict(img_path, conf=0.4)
    if result:
        result_image = cv2.cvtColor(result[0].plot(), cv2.COLOR_BGR2RGB)
        return "Segmentation completed.", result_image
    return "No segmentation detected.", None

st.title("Car Damage Assessment")
st.write("Sample image for testing: https://github.com/Shreyas95Katti/ProClaim_Capstone/blob/main/Test_Image.JPEG")
st.write("Upload an image of a car to assess its damage.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with open("uploaded_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.button("Run Assessment"):
        img_path = "uploaded_image.jpg"
        img_224 = prepare_image_224(img_path)
        img_256 = prepare_img_256(img_path)

        if pipe1(img_224, model1):
            if pipe2(img_256, model2):
                location = pipe3_loc(img_256, model3)
                severity = pipe3_sev(img_256, model4)
                object_msg, obj_image_path = object_detect(img_path)
                segmentation_msg, seg_image_path = segment(img_path)

                st.success("Assessment Completed!")
                st.write(f"**Damage Location:** {location}")
                st.write(f"**Damage Severity:** {severity}")
                st.image(obj_image_path, caption="Object Detection Result", use_container_width=True)
                st.write(f"**Segmentation:** {segmentation_msg}")
                st.image(seg_image_path, caption="Segmentation Result", use_container_width=True)

            else:
                st.warning("Damage validation failed. Please upload a clearer image.")
        else:
            st.warning("Car validation failed. Please upload a valid car image.")

import carla
import torch
import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras.models import load_model
import cv2
import numpy as np

class SteeringModel:
    def __init__(self, model_path="driving_model/steering_model.h5"):
        self.model_path = model_path
        self.model = load_model(self.model_path, compile=False)

    
    def img_preprocess(self, img, mask):
        img = img[200:, :, :]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.resize(img, (200, 66))
        img = img.astype(np.float32) / 255.0

        mask = mask[200:, :]
        mask = cv2.resize(mask, (200, 66), interpolation=cv2.INTER_NEAREST)

        return img, mask
    
    def model_predict(self, img, mask):
        img = np.expand_dims(img, axis=0)
        mask = np.expand_dims(mask, axis=0)

        steering = float(self.model.predict([img, mask], verbose=False))
        steering = np.clip(steering, -1.0, 1.0)


        return steering
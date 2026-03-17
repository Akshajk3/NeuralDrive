import torch
import cv2
import numpy as np
from driving_model.waymo.waymo_model import DrivingModel

class SteeringModel:
    def __init__(self, model_path="driving_model\waymo\waymo_model.pth"):
        self.model_path = model_path
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        
        self.model = DrivingModel(num_waypoints=20).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()

        self.seq_len = 5
        self.rgb_buffer = []
        self.mask_buffer = []
    
    def preprocess(self, img, mask=None):
        img = img[30:, :, :]
        img = (img - np.array([0.485,0.456,0.406])) / np.array([0.229,0.224,0.225])
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0

        img = np.transpose(img, (2, 0, 1))

        return img, None
    
    def waypoints_to_steering(self, waypoints):
        target = waypoints[5]

        x = target[0]
        y = target[1]

        angle = np.arctan2(x, y)

        steering = np.clip(angle, -1.0, 1.0)

        return steering
    
    def model_predict(self, img, mask=None):
        img, _ = self.preprocess(img)

        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            waypoints = self.model(img)

        waypoints = waypoints.squeeze(0).cpu().numpy()

        steering = self.waypoints_to_steering(waypoints)

        return steering
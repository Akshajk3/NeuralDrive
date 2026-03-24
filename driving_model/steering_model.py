import torch
import cv2
import numpy as np
from driving_model.waymo.model import DrivingModel
from torchvision.transforms import transforms

class SteeringModel:
    def __init__(self, model_path="driving_model/waymo/waymo_model.pth"):
        self.model_path = model_path
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            )
        ])
    
    def preprocess(self, img, mask):
        img = img[90:, :, :]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)

        return img
    
    def waypoints_to_steering(self, waypoints):
        idx = min(5, len(waypoints) - 1)
        target = waypoints[idx]

        x, y = target

        angle = np.arctan2(y, x)

        return np.clip(angle, -1.0, 1.0)

    def model_predict(self, img, mask):
        img = self.preprocess(img)

        img = img.unsqueeze(0).to(self.device)

        with torch.no_grad():
            waypoints = self.model(img)

        waypoints = waypoints.squeeze(0).cpu().numpy()

        steering = self.waypoints_to_steering(waypoints)

        return steering
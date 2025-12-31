import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
import time
from torchvision import transforms
from lane_detection.lane_detection import TUSimpleDataset, UNet

class LaneDetector:
    def __init__(self, model_path='model.h5'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

        self.model = UNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((256, 512)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print(f"Model loaded on {self.device}")

    def predict_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(frame_rgb)
        original_size = pil_image.size

        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            prediction = torch.sigmoid(output)
            mask = (prediction.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
            
        mask_pil = Image.fromarray(mask * 255)
        mask_pil = mask_pil.resize(original_size, Image.NEAREST)
        lane_mask = np.array(mask_pil) > 127

        return lane_mask
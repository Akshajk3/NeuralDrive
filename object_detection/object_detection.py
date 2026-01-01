from ultralytics import YOLO
import numpy as np
import cv2
import torch

class ObjectDetector:
    def __init__(self, yolo_model="yolov8s.pt", conf=0.4):
        self.model = YOLO(yolo_model)
        self.conf = conf

        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        self.model.to(device=self.device)

        print(f"YOLO Device: {self.model.device}")
    

    def detect_frame(self, frame):
        results = self.model(frame, conf=self.conf, verbose=False)
        annotated_frame = results[0].plot()

        return annotated_frame

    def get_distance(self, depth_map, box):
        x1, y1, x2, y2 = map(int, box)

        h, w = depth_map.shape
        x1, x2 = map(0, x1), min(w - 1, x2)
        y1, y2 = map(0, y1), min(h - 1, y2)

        region = depth_map[y1:y2, x1:x2]
        region = region[np.isfinite(region)]

        if len(region) == 0:
            return None
        
        return float(np.median(region))
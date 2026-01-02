from ultralytics import YOLO
import numpy as np
import cv2
import torch

class ObjectDetector:
    def __init__(self, yolo_model="yolo11m.pt", conf=0.4):
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
        results = self.model.track(frame, conf=self.conf, persist=True, verbose=False)
        return results

    def get_distance(self, depth_map, box):
        x1, y1, x2, y2 = map(int, box)

        h, w = depth_map.shape

        w_box, h_box = x2-x1, y2-y1
        x_center, y_center = x1 + (w_box / 2), y1 + (h_box / 2)

        scale = 0.5
        new_x1 = int(x_center - (w_box * scale) / 2)
        new_x2 = int(x_center + (w_box * scale) / 2)
        new_y1 = int(y_center - (h_box * scale) / 2)
        new_y2 = int(y_center + (h_box * scale) / 2)

        new_x1, new_x2 = max(0, new_x1), min(w, new_x2)
        new_y1, new_y2 = max(0, new_y1), min(h, new_y2)

        region = depth_map[new_y1:new_y2, new_x1:new_x2]

        region = region[np.isfinite(region)]
        region = region[(region > 0.1) & (region < 100.0)]


        if len(region) < 10:
            return None
        
        return float(np.percentile(region, 5))
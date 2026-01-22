import os
import cv2
import csv
import numpy as np
from pynput import keyboard
import carla

class SimDataLogger:
    def __init__(self, rgb_dir='sim_data/RGB', depth_dir='sim_data/depth', mask_dir='sim_data/masks', csv_file='sim_data/driving_log.csv'):
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.mask_dir = mask_dir
        self.csv_file = csv_file
        self.recording = False
        self.image_count = 0

        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(self.mask_dir, exist_ok=True)

        with open(self.csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            # writer.writerow(["rgb", "depth", "mask", "steering", "throttle", "brake", "speed"])
            writer.writerow(["rgb", "mask", "steering"])

        self.listener = keyboard.Listener(on_press=self.on_key_press)
        self.listener.start()

    def on_key_press(self, key):
        try:
            if key.char.lower() == "r":
                self.recording = not self.recording
                state = "ON" if self.recording else "OFF"
                print(f"[DataLogger] Recording {state}")
        except AttributeError:
            pass
    
    def save_rgb(self, image: np.ndarray):
        if not self.recording or image is None:
            return None
        
        filename = os.path.join(self.rgb_dir, f"{self.image_count:05d}.png")
        cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return filename

    def save_depth(self, image: np.ndarray):
        if not self.recording or image is None:
            return None

        filename = os.path.join(self.depth_dir, f"{self.image_count:05d}.png")
        cv2.imwrite(filename, image)
        return filename
    
    def save_mask(self, mask):
        if not self.recording or mask is None:
            return None
        
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        filename = os.path.join(self.mask_dir, f"{self.image_count:05d}.png")
        cv2.imwrite(filename, mask)
        return filename
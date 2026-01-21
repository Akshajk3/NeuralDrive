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
            writer.writerow(["rgb", "depth", "mask", "steering", "throttle", "brake", "speed"])

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
    
    def save_rgb(self, image: carla.Image):
        if not self.recording:
            return None
        
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, :3]
        filename = os.path.join(self.rgb_dir, f"{self.image_count:05d}.png")
        cv2.imwrite(filename, cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
        return filename

    def save_depth(self, image: carla.Image):
        if not self.recording:
            return None

        image.convert(carla.ColorConverter.LogarithmicDepth)
        depth_image = np.frombuffer(image.raw_data, dtype=np.uint8)
        depth_image = depth_image.reshape((image.height, image.width, 4))
        depth_image = depth_image[:, :, 0]

        filename = os.path.join(self.depth_dir, f"{self.image_count:05d}.png")
        cv2.imwrite(filename, depth_image)
        return filename
    
    def save_mask(self, image):
        pass
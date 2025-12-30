from lane_detection.lane_evalutation import LaneDetector
import carla
import time
from sim.observer import latest_rgb_frame
import numpy as np
import cv2


def detect_lane(image: carla.Image) -> np.ndarray:
    print ("Starting Lane Detection...")

    cv2.imshow("Latset RGB Image", latest_rgb_frame)
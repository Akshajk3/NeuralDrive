from lane_detection.lane_evalutation import LaneDetector
import carla
import time
import sim.observer as observer
import numpy as np
import cv2

lane_detector = LaneDetector('lane_detection/model.h5')

def detect_lane() -> np.ndarray | None:
    if observer.latest_rgb_frame is None:
        return None

    frame = observer.latest_rgb_frame.copy()
    lane_mask = lane_detector.predict_frame(frame)

    overlay = frame.copy()
    overlay[lane_mask] = [0, 0, 255]

    cv2.imshow("Lane Detection", overlay)
    cv2.waitKey(1)
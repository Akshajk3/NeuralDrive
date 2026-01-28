from lane_detection.lane_detector import LaneDetector
from object_detection.object_detection import ObjectDetector
from object_detection.distance_smoother import DistanceSmoother
import carla
import time
import sim.observer as observer
import numpy as np
import cv2
import random

lane_detector = LaneDetector('lane_detection/model.h5')
object_detector = ObjectDetector()
smoother = DistanceSmoother()

def detect_lane(frame) -> np.ndarray | None:
    lane_mask = lane_detector.predict_frame(frame)

    if lane_mask is None:
        return None
    
    lane_mask = (lane_mask.astype(np.uint8)) * 255

    return lane_mask

def draw_lane_mask(display_frame, lane_mask):
    display_frame[lane_mask > 0] = [0, 0, 255]



def detect_objects(frame, depth_map, display_frame):
    results = object_detector.detect_frame(frame)
    result = results[0]

    if result.boxes is None or result.boxes.id is None:
        return

    for box, cls, track_id in zip(
        result.boxes.xyxy.cpu().numpy(),
        result.boxes.cls.cpu().numpy(),
        result.boxes.id.int().cpu().numpy()
    ):
        raw_distance = object_detector.get_distance(depth_map, box)

        distance = smoother.update(track_id, raw_distance)

        x1, y1, x2, y2 = map(int, box)
        label = object_detector.model.names[int(cls)]

        text = f"{label} {distance:.1f}m" if distance else f"{label} ?m"

        cv2.putText(
            display_frame,
            text,
            (x1, y1-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 125, 0),
            2
        )

        cv2.rectangle(
            display_frame,
            (x1, y1),
            (x2, y2),
            (0, 125, 0),
            2
        )

def detect_lanes_and_objects():
    if observer.latest_rgb_frame["center"] is None or observer.latest_depth_mask is None:
        return
    
    display_frame = observer.latest_rgb_frame["center"].copy()
    lane_mask = detect_lane(observer.latest_rgb_frame["center"])

    if lane_mask is not None:
        draw_lane_mask(display_frame, lane_mask)

    detect_objects(observer.latest_rgb_frame["center"], observer.latest_depth_mask, display_frame)

    cv2.imshow("Lane and Object Detection", display_frame)
    cv2.waitKey(1)

    return lane_mask
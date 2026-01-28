import carla
import random
import time
import math
import numpy as np
import cv2

latest_rgb_frame = {
    "left": None,
    "center": None,
    "right": None
}

latest_depth_mask = None

def get_speed_kmh(vehicle: carla.Actor):
    v = vehicle.get_velocity()
    return 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z ** 2)

def get_depth_image(image: carla.Image):
    copy: carla.Image = image
    copy.convert(carla.ColorConverter.LogarithmicDepth)
    depth_image = np.frombuffer(image.raw_data, dtype=np.uint8)
    depth_image = depth_image.reshape((image.height, image.width, 4))
    depth_image = depth_image[:, :, 0]

    return depth_image

def depth_to_meters(image: carla.Image):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))
    R = arr[:, :, 0].astype(np.uint32)
    G = arr[:, :, 1].astype(np.uint32)
    B = arr[:, :, 2].astype(np.uint32)
    normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
    meters = 1000 * normalized
    return meters

def depth_callback(image: carla.Image):
    global latest_depth_mask
    latest_depth_mask = depth_to_meters(image)

    vis = get_depth_image(image)

    cv2.imshow("Depth Camera", vis)
    cv2.waitKey(1)

def get_rgb_image(image: carla.Image):
    img = np.frombuffer(image.raw_data, dtype=np.uint8)
    img = img.reshape((image.height, image.width, 4))

    img = img[:, :, :3]

    return img

def rgb_callback(index: str):
    def callback(image: carla.Image):
        img = get_rgb_image(image)

        global latest_rgb_frame
        latest_rgb_frame[index] = img

    return callback
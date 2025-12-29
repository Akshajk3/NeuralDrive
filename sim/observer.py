import carla
import random
import time
import math
import numpy as np
import cv2

def get_speed_kmh(vehicle: carla.Actor):
    v = vehicle.get_velocity()
    return 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z ** 2)

def get_depth_image(image: carla.Image):
    image.convert(carla.ColorConverter.LogarithmicDepth)
    depth = np.frombuffer(image.raw_data, dtype=np.uint8)
    depth = depth.reshape((image.height, image.width, 4))
    depth = depth[:, :, 0]

    return depth

def depth_callback(image: carla.Image):
    depth_img = get_depth_image(image)

    cv2.imshow("Depth Camera", depth_img)
    cv2.waitKey(1)

def get_rgb_image(image: carla.Image):
    img = np.frombuffer(image.raw_data, dtype=np.uint8)
    img = img.reshape((image.height, image.width, 4))

    img = img[:, :, :3]

    return img

def rgb_callback(image: carla.Image):
    img = get_rgb_image(image)

    cv2.imshow("RGB Camera", img)
    cv2.waitKey(1)

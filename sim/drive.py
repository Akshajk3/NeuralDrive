from driving_model.steering_model import SteeringModel
import carla
import numpy as np
import cv2

class Drive:
    def __init__(self):
        self.driver = SteeringModel()
        self.prev_steer = 0.0

    def compute_control(self, rgb, mask):
        img, mask = self.driver.preprocess(rgb, mask)

        steer = self.driver.model_predict(img, mask)

        steer = 0.7 * self.prev_steer + 0.3 * steer
        self.prev_steer = steer

        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = 0.4
        control.brake = 0.0

        return control
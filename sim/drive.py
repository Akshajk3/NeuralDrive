from driving_model.waymo.waymo_steering_model import SteeringModel
import carla
import numpy as np
import cv2

class Drive:
    def __init__(self):
        self.driver = SteeringModel()
        self.prev_steer = 0.0

    def compute_control(self, rgb, mask=None):
        steer = self.driver.model_predict(rgb, mask)

        # simple low-pass filter
        steer = 0.7 * self.prev_steer + 0.3 * steer
        self.prev_steer = steer

        control = carla.VehicleControl()
        control.steer = float(steer)
        control.throttle = 0.4
        control.brake = 0.0
        return control
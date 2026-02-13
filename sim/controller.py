import pygame
import carla
from data_logger.sim.sim_data_logger import SimDataLogger

class Controller:
    def __init__(self, vehicle: carla.Vehicle):
        self.vehicle = vehicle
        self.control = carla.VehicleControl()
        self.steer_cache = 0.0

    def parse_input(self, logger: SimDataLogger):
        keys = pygame.key.get_pressed()

        if keys[pygame.K_w]:
            self.control.throttle = min(self.control.throttle + 0.05, 0.35)
        else:
            self.control.throttle = 0.0

        if keys[pygame.K_s]:
            self.control.brake = max(self.control.brake + 0.2, 0.0)
        else:
            self.control.brake = 0.0

        if keys[pygame.K_a]:
            self.steer_cache -= 0.04
        elif keys[pygame.K_d]:
            self.steer_cache += 0.04
        else:
            self.steer_cache *= 0.9

        if keys[pygame.K_r]:
            logger.on_key_press()
        
        self.steer_cache = max(-1.0, min(1.0, self.steer_cache))
        self.control.steer = round(self.steer_cache, 3)

        self.vehicle.apply_control(self.control)
        return self.control
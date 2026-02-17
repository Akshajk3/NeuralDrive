import pygame
import carla
from data_logger.sim.sim_data_logger import SimDataLogger

class Controller:
    def __init__(self, vehicle: carla.Vehicle):
        self.vehicle = vehicle
        self.control = carla.VehicleControl()
        self.steer_cache = 0.0

    def parse_input(self, logger: SimDataLogger):
        for event in pygame.event.get():

            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_w:
                    self.control.throttle = 0.35

                if event.key == pygame.K_s:
                    self.control.brake = 0.5

                if event.key == pygame.K_a:
                    self.steer_cache -= 0.2

                if event.key == pygame.K_d:
                    self.steer_cache += 0.2

                if event.key == pygame.K_r:
                    logger.on_key_press()

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_w:
                    self.control.throttle = 0.0

                if event.key == pygame.K_s:
                    self.control.brake = 0.0

        self.steer_cache = max(-1.0, min(1.0, self.steer_cache))
        self.control.steer = round(self.steer_cache, 3)

        self.vehicle.apply_control(self.control)
        return self.control

import carla
import random
import time
import math
import numpy as np
import cv2

client = carla.Client("localhost", 2000)
client.set_timeout(15.0)

print("Loading Map...")
world = client.load_world("Town03")
blueprint_library = world.get_blueprint_library()

vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
spawn_point = random.choice(world.get_map().get_spawn_points())

vehicle = world.spawn_actor(vehicle_bp, spawn_point)

npc_vehicles = []
spawn_points = world.get_map().get_spawn_points()

depth_bp = blueprint_library.find("sensor.camera.depth")
depth_bp.set_attribute("image_size_x", "224")
depth_bp.set_attribute("image_size_y", "224")
depth_bp.set_attribute("fov", "90")

camera_bp = blueprint_library.find("sensor.camera.rgb")
camera_bp.set_attribute("image_size_x", "224")
camera_bp.set_attribute("image_size_y", "224")
camera_bp.set_attribute("fov", "90")

depth_transform = carla.Transform(
    carla.Location(x=1.5, z=2.2),
    carla.Rotation(pitch=0)
)

rgb_transform = carla.Transform(
    carla.Location(x=1.5, z=2.2),
    carla.Rotation(pitch=0)
)

depth_camera = world.spawn_actor(
    depth_bp,
    depth_transform,
    attach_to=vehicle
)

rgb_camera = world.spawn_actor(
    camera_bp,
    rgb_transform,
    attach_to=vehicle
)

vehicle.set_autopilot(True)

def get_speed_kmh(vehicle: carla.Actor):
    v = vehicle.get_velocity()
    return 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z ** 2)

def depth_to_meters(image: carla.Image) -> np.ndarray:
    data = np.frombuffer(image.raw_data, dtype=np.uint8)
    data = data.reshape((image.height, image.width, 4))

    R = data[:, :, 2].astype(np.float32)
    G = data[:, :, 1].astype(np.float32)
    B = data[:, :, 0].astype(np.float32)

    normalized_depth = (R * 255 * 255 + G * 255 + B) / (255**3 - 1)
    depth_meters = 1000.0 * normalized_depth
    return depth_meters

def depth_callback(image: carla.Image):
    depth_m = depth_to_meters(image)
    
    depth_vis = np.clip(depth_m / 50.0, 0, 1)
    depth_vis = (depth_vis * 255).astype(np.uint8)

    cv2.imshow("Depth Camera", image)
    cv2.waitKey(1)

depth_camera.listen(depth_callback)

print("Observing Vehicle...")

try:
    while True:
        control = vehicle.get_control()
        speed = get_speed_kmh(vehicle)

        print(
            f"Speed: {speed:6.2f} km/h | "
            f"Steer: {control.steer: .3f} | "
            f"Throttle: {control.throttle: .3f} | "
            f"Brake: {control.brake: .3f}"
        )

        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    vehicle.destroy()
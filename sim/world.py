import carla
import time
import random

from sim.observer import get_speed_kmh, depth_callback, rgb_callback
from sim.detector import detect_lanes_and_objects
from sim.npc_manager import NPCManager

client = carla.Client("localhost", 2000)
client.set_timeout(30.0)

print("Loading Map...")

world = client.load_world("Town03")
blueprint_library = world.get_blueprint_library()

vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
spawn_point = random.choice(world.get_map().get_spawn_points())

vehicle = world.spawn_actor(vehicle_bp, spawn_point)

npc_manager = NPCManager(client, world, blueprint_library)

npc_vehicles = npc_manager.spawn_npc_vehicles(200)
npc_walkers = npc_manager.spawn_npc_pedestrians(100)

depth_bp = blueprint_library.find("sensor.camera.depth")
depth_bp.set_attribute("image_size_x", "1280")
depth_bp.set_attribute("image_size_y", "720")
depth_bp.set_attribute("fov", "86")

camera_bp = blueprint_library.find("sensor.camera.rgb")
camera_bp.set_attribute("image_size_x", "1280")
camera_bp.set_attribute("image_size_y", "720")
camera_bp.set_attribute("fov", "86")

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

depth_camera.listen(depth_callback)
rgb_camera.listen(rgb_callback)

vehicle.set_autopilot(True)

def run_sim():
    print("Running Sim...")
    try:
        while True:
            world.tick()
            control = vehicle.get_control()
            speed = get_speed_kmh(vehicle)

            print(
                f"Speed: {speed:6.2f} km/h | "
                f"Steer: {control.steer: .3f} | "
                f"Throttle: {control.throttle: .3f} | "
                f"Brake: {control.brake: .3f}"
            )

            detect_lanes_and_objects()
            
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        for npc in npc_vehicles:
            npc.destroy()

        depth_camera.stop()
        rgb_camera.stop()

        depth_camera.destroy()
        rgb_camera.destroy()

        vehicle.destroy()
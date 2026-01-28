import carla
import random
import csv
import cv2

import sim.observer as observer
from sim.observer import get_speed_kmh, depth_callback, rgb_callback
from sim.detector import detect_lanes_and_objects
from sim.npc_manager import NPCManager
from data_logger.sim.sim_data_logger import SimDataLogger
from sim.drive import Drive

client = carla.Client("localhost", 2000)
client.set_timeout(30.0)

print("Loading Map...")

world = client.load_world("Town04")
blueprint_library = world.get_blueprint_library()

vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
spawn_point = random.choice(world.get_map().get_spawn_points())

vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# npc_manager = NPCManager(client, world, blueprint_library)

# npc_vehicles = npc_manager.spawn_npc_vehicles(200)
# npc_walkers = npc_manager.spawn_npc_pedestrians(100)

data_logger = SimDataLogger()
controller = Drive()

depth_bp = blueprint_library.find("sensor.camera.depth")
depth_bp.set_attribute("image_size_x", "640")
depth_bp.set_attribute("image_size_y", "360")
depth_bp.set_attribute("fov", "86")

camera_bp = blueprint_library.find("sensor.camera.rgb")
camera_bp.set_attribute("image_size_x", "640")
camera_bp.set_attribute("image_size_y", "360")
camera_bp.set_attribute("fov", "86")

depth_transform = carla.Transform(
    carla.Location(x=1.5, z=2.2),
    carla.Rotation(pitch=0)
)

rgb_transform_left = carla.Transform(
    carla.Location(x=1.5, y=-0.35, z=2.2),
    carla.Rotation(pitch=0)
)

rgb_transform_center = carla.Transform(
    carla.Location(x=1.5, y=0.0, z=2.2),
    carla.Rotation(pitch=0)
)

rgb_transform_right = carla.Transform(
    carla.Location(x=1.5, y=0.35, z=2.2),
    carla.Rotation(pitch=0)
)

depth_camera = world.spawn_actor(
    depth_bp,
    depth_transform,
    attach_to=vehicle
)

left_camera = world.spawn_actor(
    camera_bp,
    rgb_transform_left,
    attach_to=vehicle
)

left_camera = world.spawn_actor(
    camera_bp,
    rgb_transform_left,
    attach_to=vehicle
)

center_camera = world.spawn_actor(
    camera_bp,
    rgb_transform_center,
    attach_to=vehicle
)

right_camera = world.spawn_actor(
    camera_bp,
    rgb_transform_right,
    attach_to=vehicle
)

depth_camera.listen(depth_callback)
left_camera.listen(rgb_callback(index='left'))
center_camera.listen(rgb_callback(index='center'))
right_camera.listen(rgb_callback(index='right'))


# Uncomment if you want Autopilot to run for data collection
traffic_manager = client.get_trafficmanager(8000)
traffic_manager.set_synchronous_mode(False)
traffic_manager.ignore_lights_percentage(vehicle, 100.0)
vehicle.set_autopilot(True, 8000)

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

            lane_mask = detect_lanes_and_objects()

            # rgb = observer.latest_rgb_frame["center"]

            # if rgb is not None and lane_mask is not None:
            #     control = controller.compute_control(rgb, lane_mask)
            #     vehicle.apply_control(control)

            if data_logger.recording:
                left = data_logger.save_rgb(observer.latest_rgb_frame["left"], "left")
                center = data_logger.save_rgb(observer.latest_rgb_frame["center"], "center")
                right = data_logger.save_rgb(observer.latest_rgb_frame["right"], "right")
                mask = data_logger.save_mask(lane_mask)

                cv2.imshow("left", observer.latest_rgb_frame["left"])
                cv2.imshow("center", observer.latest_rgb_frame["center"])
                cv2.imshow("right", observer.latest_rgb_frame["right"])
                
                with open(data_logger.csv_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        left,
                        center,
                        right,
                        mask,
                        control.steer
                    ])
                
                data_logger.image_count += 1

                if data_logger.image_count >= 5000:
                    break

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        # for npc in npc_vehicles:
        #     npc.destroy()

        depth_camera.stop()
        center_camera.stop()
        left_camera.stop()
        right_camera.stop()

        depth_camera.destroy()
        center_camera.destroy()
        left_camera.destroy()
        right_camera.destroy()

        vehicle.destroy()
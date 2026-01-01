import carla
import random

class NPCManager:
    def __init__(self, client: carla.Client, world: carla.World, blueprint_library: carla.BlueprintLibrary):
        self.client = client
        self.world = world
        self.bp_library = blueprint_library
        self.spawned_vehicles = []
        self.spawned_walkers = []
        self.walker_controllers = []
    
    def spawn_npc_vehicles(self, num_vehicles=20):
        traffic_manager = self.client.get_trafficmanager(8000)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        traffic_manager.set_synchronous_mode(False)
        traffic_manager.set_hybrid_physics_mode(True)
        traffic_manager.set_hybrid_physics_radius(70.0)
        
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        vehicle_bps = self.bp_library.filter("vehicle")

        for spawn in spawn_points:
            bp = random.choice(vehicle_bps)

            if bp.has_attribute("color"):
                color = random.choice(bp.get_attribute("color").recommended_values)
                bp.set_attribute("color", color)

            npc = self.world.try_spawn_actor(bp, spawn)

            if npc:
                npc.set_autopilot(True, traffic_manager.get_port())
                self.spawned_vehicles.append(npc)

        print(f"Spawned {len(self.spawned_vehicles)} NPC Vehicles")
    
    def spawn_npc_pedestrians(self, num_pedestrians=30):
        walker_blueprints = self.bp_library.filter("walker.pedestrian.*")
        controller_bp = self.bp_library.find("controller.ai.walker")
        self.world.set_pedestrians_cross_factor(0.3)

        spawn_points = []
        
        for _ in range(num_pedestrians):
            loc = self.world.get_random_location_from_navigation()
            if loc is not None:
                spawn_points.append(loc)

        for loc in spawn_points:
            spawn = carla.Transform(loc)
            walker_bp = random.choice(walker_blueprints)

            if walker_bp.has_attribute("is_invincible"):
                walker_bp.set_attribute("is_invincible", "false")

            walker = self.world.try_spawn_actor(walker_bp, spawn)
            if walker is None:
                continue

            controller = self.world.spawn_actor(controller_bp, carla.Transform(), attach_to=walker)

            controller.start()
            controller.go_to_location(self.world.get_random_location_from_navigation())
            controller.set_max_speed(random.uniform(1.2, 2.2))


            self.spawned_walkers.append(walker)
            self.walker_controllers.append(controller)

        print(f"Spawned {len(self.spawned_walkers)} pedestrians")
        return self.spawned_walkers, self.walker_controllers

    def destroy(self):
        for vehicle in self.spawned_vehicles:
            vehicle.destroy()

        for walker in self.spawned_walkers:
            walker.destroy()
import numpy as np
from . import Drone
from . import constants as C

class Swarm():
    def __init__(self, num_drones, shape="cube"):
        self.N = num_drones
        self.drones = []
        self.timestep = 0

        self.position_history = []

        if shape == "cube":
            # For now, if cube, assume num_drones has a perfect cube root.
            side_len = int(num_drones ** (1 / 3))

            for layer_number in range(side_len):
                z_loc = C.SEPARATION * layer_number
                for row in range(side_len):
                    x_loc = C.SEPARATION * row
                    for col in range(side_len):
                        y_loc = C.SEPARATION * col
                        d = Drone.Drone()
                        d.pos = np.asarray([x_loc, y_loc, z_loc])
                        d.pos_initial = d.pos
                        d.target = d.pos
                        self.drones.append(d)

        if shape == "planar":
            side_len = int(num_drones ** (1 / 2))
            self.G = np.ones((side_len ** 2, side_len ** 2), dtype=int)  # Change this to change network

            z_loc = 0
            for row in range(side_len):
                x_loc = C.SEPARATION * row
                for col in range(side_len):
                    y_loc = C.SEPARATION * col
                    d = Drone.Drone()
                    d.pos = np.asarray([x_loc, y_loc, z_loc])
                    d.pos_initial = d.pos
                    d.target = d.pos
                    self.drones.append(d)

        self.init_drone_PIDs()

    #### "Public" Methods #########
    def tick(self):
        # So we just have the swarm update drones? Maybe it tracks data? idk.
        for d in self.drones:
            d.update_state_from_pos(d.pos)

        self.update_data()
        self.timestep += 1

    def set_swarm_target_relative(self, dpos):
        delta = np.asarray(dpos)
        for d in self.drones:
            d.target = d.pos_initial + delta
        # d.init_PIDs()

    def set_swarm_pos_relative(self, dpos):
        delta = np.asarray(dpos)
        for d in self.drones:
            d.pos = d.pos_initial + delta

    # Should be called when we change the target.
    def init_drone_PIDs(self):
        for d in self.drones:
            d.init_PIDs()

    def update_data(self):
        self.position_history.append([d.pos for d in self.drones])
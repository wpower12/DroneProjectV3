import numpy as np
import math
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

from . import Drone
from . import GCRFModel as M
from . import constants as C


class Swarm():
    def __init__(self, swarm_options):
        # swarm_options = [num_drones, swarm_shape, [color_train, color_inference], start_pt, end_pt, use_structure]:
        self.N, self.shape, self.color, self.start_pt, self.end_pt, self.use_structure = swarm_options
        self.swarm_id = 0

        self.drones = []
        self.training = True
        self.use_model = True

        # Global timestep counter
        self.timestep = 0

        # Current Dataset
        self.data_window = [[] for ind in range(0, self.N)]
        self.curr_X = None
        self.curr_Y = None
        self.curr_S = None

        # Historical Data
        self.X = [None for ind in range(0, self.N)]
        self.Y = [None for ind in range(0, self.N)]
        self.S = None

        self.swarm_mean = np.zeros(3)
        self.swarm_variance = np.zeros(3)
        # GCRF Model.
        self.model = None

        # Generate the swarm by generating drones based on the desired shape
        self.generate_drones_from_shape()
        # Set drones' initial positions and targets and Init PIDs
        self.set_swarm_pos_relative(self.start_pt)
        self.set_swarm_target_relative(self.end_pt)
        self.init_drone_PIDs()

    def generate_model(self, prng, num_bagging_learners=5):
        if self.use_structure is not None:
            self.model = M.GCRFModel(M=num_bagging_learners, rnd_state=prng)
        else:
            self.model = None

    #### Swarm Generation (shape-based) #########
    def generate_drones_from_shape(self):
        if self.shape == 'cube':
            self.generate_cube_swarm()
        elif self.shape == 'planar':
            self.generate_planar_swarm()

    def generate_cube_swarm(self):
        side_len = int(self.N ** (1 / 3))
        for layer_number in range(side_len):
            z_loc = C.SEPARATION * layer_number
            for row in range(side_len):
                x_loc = C.SEPARATION * row
                for col in range(side_len):
                    y_loc = C.SEPARATION * col
                    d = Drone.Drone([x_loc, y_loc, z_loc])
                    self.drones.append(d)

    def generate_planar_swarm(self):
        side_len = int(self.N ** (1 / 2))
        z_loc = 0
        for row in range(side_len):
            x_loc = C.SEPARATION * row
            for col in range(side_len):
                y_loc = C.SEPARATION * col
                d = Drone.Drone([x_loc, y_loc, z_loc])
                self.drones.append(d)

    #### "Public" Methods #########
    def tick(self, wind):

        if self.training:
            self.training_tick()

        self.update_data()

        # INFERENCE
        if not self.training:
            self.inference_tick()

        # Data Gathering/Model Training
        if self.use_structure is not None and self.training:

            if self.timestep >= C.NUM_TRAINING_STEPS - 1:
                self.model.train(self.S, self.X, self.Y, self.use_structure)
            # print('theta:', self.model.theta)
            # pass

        self.timestep += 1

    def training_tick(self):
        for d in self.drones:
            d.update_training()

    def inference_tick(self):
        if self.use_structure is not None:
            pos_estimates, pos_variances = self.model.predict(self.curr_X, self.curr_S, self.use_structure)

        for index, d in enumerate(self.drones):

            if self.use_structure is not None:
                d.pos_estimate = pos_estimates[index, :]
                d.pos_variance = pos_variances[index, :]

            # Velocity included in estimate only when not using DR
            if self.use_structure is None:
                d.update_inference(vel_included_in_prediction=False)
            else:
                d.update_inference(vel_included_in_prediction=True)

    def move_swarm(self, shift_vector, drafting_active=True):
        if drafting_active:
            drone_multipliers = self.per_drone_wind_multipliers(shift_vector, layer_wind_multiplier=0.9)
        else:
            drone_multipliers = np.ones(self.N)

        for index, d in enumerate(self.drones):
            weights_d = np.asarray([drone_multipliers[index], drone_multipliers[index], 1])
            d.move_drone(shift_vector, weights_d)

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

    def set_swarm_id(self, id):
        self.swarm_id = id
        for d in self.drones:
            d.swarm_index = id

    ######################

    #### Model Methods ###########
    def get_curr_S(self):
        curr_S = np.zeros((self.N, self.N), dtype=float)

        for i in range(self.N):
            for j in range(self.N):
                if j > i:
                    # Similarity is their distance
                    d_i = self.drones[i]
                    d_j = self.drones[j]
                    if self.training:
                        curr_S[i][j] = np.linalg.norm(d_i.pos - d_j.pos)
                    else:
                        curr_S[i][j] = np.linalg.norm(d_i.pos_estimate - d_j.pos_estimate)
                    curr_S[j][i] = curr_S[i][j]

        # Threshold S
        curr_S = curr_S / np.max(curr_S)
        curr_S = 1.0 - curr_S

        return curr_S

    def update_data(self):
        if self.timestep < C.NUM_TRAINING_STEPS:
            for ind, d in enumerate(self.drones):
                self.data_window[ind].append(np.copy(d.pos))
        else:
            for ind, d in enumerate(self.drones):
                self.data_window[ind].append(np.copy(d.pos_estimate))

        if len(self.data_window[0]) > C.WINDOW_SIZE:

            self.curr_X = np.zeros((self.N, 3 * C.WINDOW_SIZE), dtype=float)
            self.curr_Y = np.zeros((self.N, 3), dtype=float)

            self.curr_S = self.get_curr_S()
            if self.S is None:
                self.S = np.copy(self.curr_S)
            else:
                mat_right = np.zeros((self.S.shape[0], self.curr_S.shape[1]), dtype=float)
                self.S = np.hstack((self.S, mat_right))

                mat_bottom = np.zeros((self.curr_S.shape[0], self.S.shape[1]), dtype=float)
                mat_bottom[:, self.S.shape[1] - self.curr_S.shape[1]:] = np.copy(self.curr_S)

                self.S = np.vstack((self.S, mat_bottom))

            for ind in range(0, self.N):

                for k in range(0, C.WINDOW_SIZE):
                    self.curr_X[ind, k * 3] = np.copy(self.data_window[ind][k][0])
                    self.curr_X[ind, k * 3 + 1] = np.copy(self.data_window[ind][k][1])
                    self.curr_X[ind, k * 3 + 2] = np.copy(self.data_window[ind][k][2])

                for i in range(0, self.N):
                    self.curr_Y[ind, 0] = np.copy(self.data_window[ind][-1][0])
                    self.curr_Y[ind, 1] = np.copy(self.data_window[ind][-1][1])
                    self.curr_Y[ind, 2] = np.copy(self.data_window[ind][-1][2])

                '''
                if (self.timestep - C.WINDOW_SIZE) < self.T_train:
                    self.X[ind][self.timestep - C.WINDOW_SIZE, :] = np.copy(self.curr_X[ind, :])
                    self.Y[ind][self.timestep - C.WINDOW_SIZE, :] = np.copy(self.curr_Y[ind, :])
                    self.S[:, :, ] = np.copy(self.curr_S)
                    t = self.timestep - C.WINDOW_SIZE
                    [t*self.N:t*self.N+self.N, t*self.N:t*self.N+self.N] = 
                    
                else:
                    self.X[ind] = np.vstack( (self.X[ind], self.curr_X[ind, :]) )
                    self.Y[ind] = np.vstack( (self.Y[ind], self.curr_Y[ind, :]) )
                        
                    self.S[:, :, self.timestep - C.WINDOW_SIZE] = np.copy(self.curr_S)
                '''

                if self.X[ind] is None:
                    self.X[ind] = np.copy(self.curr_X[ind, :])
                    self.Y[ind] = np.copy(self.curr_Y[ind, :])
                else:
                    self.X[ind] = np.vstack((self.X[ind], self.curr_X[ind, :]))
                    self.Y[ind] = np.vstack((self.Y[ind], self.curr_Y[ind, :]))

                self.data_window[ind].pop(0)

    ######################
    def dump_state(self):
        print(np.shape(self.data_x), np.shape(self.hdata_x))

    def dump_locations(self, return_true_positions):
        if return_true_positions:
            return np.asarray([d.pos for d in self.drones])
        else:
            return np.asarray([d.pos_estimate for d in self.drones])

    def per_drone_wind_multipliers(self, wind_vector, layer_wind_multiplier=0.9):
        drone_position_matrix = np.asarray([[d.pos[0], d.pos[1]] for d in self.drones])
        wind_vec_orth = np.asarray([-wind_vector[1], wind_vector[0]])
        no_drones_covered = 0
        no_exposed_current_layer = 0
        wind_fact = 1

        drone_position_matrix = np.append(drone_position_matrix, np.zeros(self.N).reshape(self.N, 1), axis=1)
        drone_position_matrix = np.append(drone_position_matrix, np.arange(self.N).reshape(self.N, 1), axis=1)
        # print (self.wind_dev)
        while (self.N - no_drones_covered) > 2:
            reached_edges = [False, False]
            exposed_hull = []
            no_exposed_current_layer = 0

            hull = ConvexHull(drone_position_matrix[:self.N - no_drones_covered, 0:2])
            projections = np.matmul(drone_position_matrix[hull.vertices, 0:2], wind_vector[:2]) / np.linalg.norm(wind_vector[:2])
            projections_orth = np.matmul(drone_position_matrix[hull.vertices, 0:2], wind_vec_orth) / np.linalg.norm(wind_vec_orth)

            sorted_proj_indexes = np.argsort(projections)
            sorted_proj_orth_indexes = np.argsort(projections_orth)

            for i in sorted_proj_indexes:
                drone_position_matrix[hull.vertices[i], 2] = wind_fact
                exposed_hull.append(i)
                no_exposed_current_layer += 1
                if hull.vertices[i] == hull.vertices[sorted_proj_orth_indexes[0]]:
                    reached_edges[0] = True
                elif hull.vertices[i] == hull.vertices[sorted_proj_orth_indexes[-1]]:
                    reached_edges[1] = True

                if reached_edges[0] and reached_edges[1]:
                    break

            sorted_indexes = np.sort(hull.vertices[exposed_hull])
            for i in range(1, no_exposed_current_layer + 1):
                row = sorted_indexes[-i]
                drone_position_matrix[[self.N - no_drones_covered - i, row]] = drone_position_matrix[[row, self.N - no_drones_covered - i]]

            no_drones_covered += no_exposed_current_layer
            wind_fact = wind_fact * layer_wind_multiplier

        # All that's left is to check if we have one or two points left and fill that with the next wind_fact
        if (self.N - no_drones_covered) >= 1:
            drone_position_matrix[0, 2] = wind_fact
        if (self.N - no_drones_covered) == 2:
            drone_position_matrix[1, 2] = wind_fact

        return drone_position_matrix[np.argsort(drone_position_matrix[:, 3])][:, 2]

    def calculateSwarmVariance2(self):
        self.updateSwarmMean()

        if self.training:
            return np.zeros(3)  # should we return a variance that includes all drones?

        xyz_min = None
        xyz_max = None

        for i, d in enumerate(self.drones):
            x, y, z = d.pos_estimate
            x_var, y_var, z_var = d.pos_variance
            if not (x_var > 0 or y_var > 0 or z_var > 0):
                continue

            if xyz_min is None:
                xyz_min = np.reshape(np.asarray([x - 3 * x_var, y - 3 * y_var, z - 3 * z_var]), (1, 3, 1))
                xyz_max = np.reshape(np.asarray([x + 3 * x_var, y + 3 * y_var, z + 3 * z_var]), (1, 3, 1))

                xyz_min = np.concatenate((xyz_min, np.reshape(np.asarray([i, i, i]), (1, 3, 1))), axis=2)
                xyz_max = np.concatenate((xyz_max, np.reshape(np.asarray([i, i, i]), (1, 3, 1))), axis=2)
            else:
                curr_min = np.reshape(np.asarray([x - 3 * x_var, y - 3 * y_var, z - 3 * z_var]), (1, 3, 1))
                curr_min = np.concatenate((curr_min, np.reshape(np.asarray([i, i, i]), (1, 3, 1))), axis=2)

                curr_max = np.reshape(np.asarray([x + 3 * x_var, y + 3 * y_var, z + 3 * z_var]), (1, 3, 1))
                curr_max = np.concatenate((curr_max, np.reshape(np.asarray([i, i, i]), (1, 3, 1))), axis=2)

                xyz_min = np.concatenate((xyz_min, curr_min), axis=0)
                xyz_max = np.concatenate((xyz_max, curr_max), axis=0)

                min_index = np.argmin(xyz_min, axis=0)
                max_index = np.argmax(xyz_max, axis=0)

                xyz_min = np.reshape(xyz_min[min_index[:, 0], range(3), :], (1, 3, 2))
                xyz_max = np.reshape(xyz_max[max_index[:, 0], range(3), :], (1, 3, 2))

        if xyz_min is None:
            return np.zeros(3)

        distances = np.vstack((self.swarm_mean - xyz_min[0, :, 0], self.swarm_mean - xyz_max[0, :, 0]))
        distances = np.absolute(distances)
        distances = np.amax(distances, axis=0)

        return distances

    def updateSwarmMean(self):
        self.swarm_mean = np.zeros(3)
        for d in self.drones:
            if self.training:
                self.swarm_mean += d.pos
            else:
                self.swarm_mean += d.pos_estimate
        self.swarm_mean /= self.N

    def updateSwarmMeanPostCollisionAvoidance(self):
        self.swarm_mean = np.zeros(3)
        for d in self.drones:
            if self.training:
                self.swarm_mean += d.pos
            else:
                self.swarm_mean += d.pos_estimate_animate
        self.swarm_mean /= self.N

    def calculateSwarmVariance3(self):
        self.updateSwarmMeanPostCollisionAvoidance()

        dists = np.zeros((self.N,), dtype=float)
        for i, d in enumerate(self.drones):
            dists[i] = np.linalg.norm(self.swarm_mean - d.pos_estimate_animate)

        max_index = np.argmax(dists)
        swarm_var = dists[max_index] + np.max(self.drones[max_index].pos_variance)

        self.swarm_variance = np.asarray([swarm_var, swarm_var, swarm_var])
        return self.swarm_variance

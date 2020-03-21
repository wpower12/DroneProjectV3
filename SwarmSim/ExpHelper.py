import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

def calc_mse(target, observed):
    '''
	n, d = np.shape(target)
	diff = target-observed
	ret = 0
	for i in range(n):
		ret += np.linalg.norm(diff[i])
	return ret
	'''
    return np.array([mean_squared_error(target[:, 0], observed[:, 0]),
                     mean_squared_error(target[:, 1], observed[:, 1]),
                     mean_squared_error(target[:, 2], observed[:, 2])])


def calc_r2_score(target, observed):
    return np.array([r2_score(target[:, 0], observed[:, 0]),
                     r2_score(target[:, 1], observed[:, 1]),
                     r2_score(target[:, 2], observed[:, 2])])


def return_wind_dev_multipliers(drone_position_matrix, wind_vector, layer_wind_multiplier=0.9):
    no_drones, _ = drone_position_matrix.shape
    wind_vec_orth = np.asarray([-wind_vector[1], wind_vector[0]])
    no_drones_covered = 0
    no_exposed_current_layer = 0
    wind_fact = 1
    wind_weight = 0.9

    np.append(drone_position_matrix, np.zeros(no_drones).reshape(no_drones, 1), axis=1)
    np.append(drone_position_matrix, np.arange(no_drones).reshape(no_drones, 1), axis=1)

    while (no_drones - no_drones_covered) > 2:
        reached_edges = [False, False]
        exposed_hull = []
        no_exposed_current_layer = 0

        hull = ConvexHull(drone_position_matrix[:no_drones - no_drones_covered, 0:2])
        projections = np.matmul(drone_position_matrix[hull.vertices, 0:2], wind_vector) / np.linalg.norm(wind_vector)
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
            drone_position_matrix[[no_drones - no_drones_covered - i, row]] = drone_position_matrix[[row, no_drones - no_drones_covered - i]]

        no_drones_covered += no_exposed_current_layer
        wind_fact = wind_fact * wind_weight

    # All that's left is to check if we have one or two points left and fill that with the next wind_fact
    if (no_drones - no_drones_covered) >= 1:
        drone_position_matrix[0, 2] = wind_fact
    if (no_drones - no_drones_covered) == 2:
        drone_position_matrix[1, 2] = wind_fact

    return drone_position_matrix[np.argsort(drone_position_matrix[:, 3])][:, 2]

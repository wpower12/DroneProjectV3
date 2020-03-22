import numpy as np
from . import constants as C


class PositionsHistoryDataFrame():
    def __init__(self, num_drones):
        # Current Dataset
        self.N = num_drones
        self.data_window = [[] for ind in range(0, self.N)]
        self.curr_X = None
        self.curr_Y = None
        self.curr_S = None

        # Historical Data
        self.X = [None for ind in range(0, self.N)]
        self.Y = [None for ind in range(0, self.N)]
        self.S = None

    def window_full(self):
        if len(self.data_window[0]) > C.WINDOW_SIZE:
            return True
        else:
            return False

    def get_curr_S(self, drones_positions):
        curr_S = np.zeros((self.N, self.N), dtype=float)

        for i in range(self.N):
            for j in range(self.N):
                if j > i:
                    # Similarity is their distance
                    curr_S[i][j] = np.linalg.norm(drones_positions[i] - drones_positions[j])
                    curr_S[j][i] = curr_S[i][j]

        # Threshold S
        curr_S = curr_S / np.max(curr_S)
        curr_S = 1.0 - curr_S

        return curr_S

    def reset_current_states(self):
        self.curr_X = np.zeros((self.N, 3 * C.WINDOW_SIZE), dtype=float)
        self.curr_Y = np.zeros((self.N, 3), dtype=float)

    def append_current_similarity_mat(self, drones_positions):
        self.curr_S = self.get_curr_S(drones_positions=drones_positions)
        if self.S is None:
            self.S = np.copy(self.curr_S)
        else:
            mat_right = np.zeros((self.S.shape[0], self.curr_S.shape[1]), dtype=float)
            self.S = np.hstack((self.S, mat_right))

            mat_bottom = np.zeros((self.curr_S.shape[0], self.S.shape[1]), dtype=float)
            mat_bottom[:, self.S.shape[1] - self.curr_S.shape[1]:] = np.copy(self.curr_S)

            self.S = np.vstack((self.S, mat_bottom))

    def set_current_states(self):
        for ind in range(0, self.N):

            for k in range(0, C.WINDOW_SIZE):
                self.curr_X[ind, k * 3] = np.copy(self.data_window[ind][k][0])
                self.curr_X[ind, k * 3 + 1] = np.copy(self.data_window[ind][k][1])
                self.curr_X[ind, k * 3 + 2] = np.copy(self.data_window[ind][k][2])

            for i in range(0, self.N):
                self.curr_Y[ind, 0] = np.copy(self.data_window[ind][-1][0])
                self.curr_Y[ind, 1] = np.copy(self.data_window[ind][-1][1])
                self.curr_Y[ind, 2] = np.copy(self.data_window[ind][-1][2])

            # Move window a step forward
            self.data_window[ind].pop(0)

    def add_current_states_to_train_data(self):
        for ind in range(0, self.N):
            if self.X[ind] is None:
                self.X[ind] = np.copy(self.curr_X[ind, :])
                self.Y[ind] = np.copy(self.curr_Y[ind, :])
            else:
                self.X[ind] = np.vstack((self.X[ind], self.curr_X[ind, :]))
                self.Y[ind] = np.vstack((self.Y[ind], self.curr_Y[ind, :]))

    def update_data(self, drones_positions):
        for i, dpos in enumerate(drones_positions):
            self.data_window[i].append(np.copy(dpos))

        if self.window_full():
            self.reset_current_states()
            self.append_current_similarity_mat(drones_positions)

            self.set_current_states()
            self.add_current_states_to_train_data()

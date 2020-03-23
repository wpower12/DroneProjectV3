import numpy as np
from simple_pid import PID
from . import constants as C


class Drone():
    u = np.linspace(0, 2 * np.pi, 20)  # 100)
    v = np.linspace(0, np.pi, 20)  # 100)
    var_x_factor = np.outer(np.cos(u), np.sin(v))
    var_y_factor = np.outer(np.sin(u), np.sin(v))
    var_z_factor = np.outer(np.ones(np.size(u)), np.cos(v))

    def __init__(self, start_position=None):
        # State Vectors
        if start_position is None:
            self.pos = np.zeros((3))  # ACTUAL Location
            self.pos_initial = np.zeros((3))  # Saved positions
            self.target = np.zeros((3))  # Actual position target used for setpoints
        else:
            self.pos = np.asarray(start_position)
            self.pos_initial = self.pos
            self.target = self.pos

        self.pos_estimate = np.zeros((3))  # Model estimate in inference
        self.pos_variance = np.zeros((3))

        self.vel = np.zeros((3))  # Velocity based on self.pos
        self.vel_estimate = np.zeros((3))  # Velocity based on self.pos_estimate

        self.acc = np.zeros((3))  # Acceleration based on self.pos
        self.acc_estimate = np.zeros((3))  # Acceleration based on self.pos_estimate

        # Controllers
        self.PID_X = None
        self.PID_Y = None
        self.PID_Z = None

        self.PID_X_estimate = None
        self.PID_Y_estimate = None
        self.PID_Z_estimate = None

        # History Lists
        self.H_pos = []
        self.H_pos_estimate = []
        self.swarm_index = None

        # Drawing colors
        self.train_color = 'b'
        self.inference_color = 'r'

    def move_drone(self, shift_vec, weights=None):
        if weights is None:
            self.pos += np.asarray([shift_vec[0], shift_vec[1], shift_vec[2]])
        else:
            self.pos += np.asarray([shift_vec[0] * weights[0], shift_vec[1] * weights[1], shift_vec[2] * weights[2]])

    def set_target(self, t):
        self.target = np.copy(t)
        self.init_PIDs()

    def init_PIDs(self):
        # Assuming we call this everytime we update the targets
        x, y, z = self.target
        self.PID_X = PID(C.PID_P, C.PID_I, C.PID_D, setpoint=x, sample_time=None)
        self.PID_Y = PID(C.PID_P, C.PID_I, C.PID_D, setpoint=y, sample_time=None)
        self.PID_Z = PID(C.PID_P, C.PID_I, C.PID_D, setpoint=z, sample_time=None)

        self.PID_X_estimate = PID(C.PID_P, C.PID_I, C.PID_D, setpoint=x, sample_time=None)
        self.PID_Y_estimate = PID(C.PID_P, C.PID_I, C.PID_D, setpoint=y, sample_time=None)
        self.PID_Z_estimate = PID(C.PID_P, C.PID_I, C.PID_D, setpoint=z, sample_time=None)

    def update_state_from_pos(self):
        # Update output limits manually so that we wouldn't clamp the last integral and the last output value
        self.PID_X.output_limits = (-1.0 * C.MAX_ACC - self.acc[0], C.MAX_ACC - self.acc[0])
        self.PID_Y.output_limits = (-1.0 * C.MAX_ACC - self.acc[1], C.MAX_ACC - self.acc[1])
        self.PID_Z.output_limits = (-1.0 * C.MAX_ACC - self.acc[2], C.MAX_ACC - self.acc[2])

        # Changes in acc are the outputs of the PID controllers
        dAcc_x = self.PID_X(self.pos[0], dt=C.DT)
        dAcc_y = self.PID_Y(self.pos[1], dt=C.DT)
        dAcc_z = self.PID_Z(self.pos[2], dt=C.DT)

        # Update acc's by clamp adding differences in acceleration (previously clamped by setting output_limits)
        self.acc = np.asarray([self.acc[0] + dAcc_x, self.acc[1] + dAcc_y, self.acc[2] + dAcc_z])

        # Update velocities by clamp adding the velocity contributions obtained from accelerating DT 'seconds'.
        n_vel_x = clamp_add(self.vel[0], self.acc[0] * C.DT, C.MAX_VEL)
        n_vel_y = clamp_add(self.vel[1], self.acc[1] * C.DT, C.MAX_VEL)
        n_vel_z = clamp_add(self.vel[2], self.acc[2] * C.DT, C.MAX_VEL)

        # self.vel = np.asarray([n_vel_x, n_vel_y, n_vel_z])
        # self.vel /= np.linalg.norm(self.vel)

        if self.swarm_index == 1:
            self.vel = np.asarray([1.0, 1.0, 1.0])
        else:
            self.vel = -np.asarray([1.0, 1.0, 1.0])

    def update_state_from_pos_estimate(self):
        # Update output limits manually so that we wouldn't clamp the last integral and the last output value
        self.PID_X_estimate.output_limits = (-1.0 * C.MAX_ACC - self.acc_estimate[0], C.MAX_ACC - self.acc_estimate[0])
        self.PID_Y_estimate.output_limits = (-1.0 * C.MAX_ACC - self.acc_estimate[1], C.MAX_ACC - self.acc_estimate[1])
        self.PID_Z_estimate.output_limits = (-1.0 * C.MAX_ACC - self.acc_estimate[2], C.MAX_ACC - self.acc_estimate[2])

        # Changes in acc are the outputs of the PID controllers
        dAcc_x = self.PID_X_estimate(self.pos_estimate[0], dt=C.DT)
        dAcc_y = self.PID_Y_estimate(self.pos_estimate[1], dt=C.DT)
        dAcc_z = self.PID_Z_estimate(self.pos_estimate[2], dt=C.DT)

        # Update acc's by clamp adding differences in acceleration (previously clamped by setting output_limits)
        self.acc_estimate = np.asarray(
            [self.acc_estimate[0] + dAcc_x, self.acc_estimate[1] + dAcc_y, self.acc_estimate[2] + dAcc_z])

        # Update velocities by clamp adding the velocity contributions obtained from accelerating DT 'seconds'.
        n_vel_x = clamp_add(self.vel_estimate[0], self.acc_estimate[0] * C.DT, C.MAX_VEL)
        n_vel_y = clamp_add(self.vel_estimate[1], self.acc_estimate[1] * C.DT, C.MAX_VEL)
        n_vel_z = clamp_add(self.vel_estimate[2], self.acc_estimate[2] * C.DT, C.MAX_VEL)

        # self.vel_estimate = np.asarray([n_vel_x, n_vel_y, n_vel_z])
        # self.vel_estimate /= np.linalg.norm(self.vel_estimate)

        if self.swarm_index == 1:
            self.vel = np.asarray([1.0, 1.0, 1.0])
        else:
            self.vel = -np.asarray([1.0, 1.0, 1.0])

    def update_training(self):
        self.update_state_from_pos()
        self.pos_estimate = np.copy(self.pos)

        self.pos += self.vel * C.DT
        self.H_pos.append(np.copy(self.pos))

        self.update_state_from_pos_estimate()
        self.pos_estimate += self.vel_estimate * C.DT

        #self.H_pos_estimate.append(np.copy(self.pos_estimate)) # Don't want to draw the whole path in inference

    def update_inference(self, vel_included_in_prediction=False):
        # Update ground truth
        self.update_state_from_pos()
        self.pos += self.vel * C.DT
        self.H_pos.append(np.copy(self.pos))

        # Move predicted positions (usually only when DR is used)
        if not vel_included_in_prediction:
            self.update_state_from_pos_estimate()
            self.pos_estimate += self.vel_estimate * C.DT

        self.H_pos_estimate.append(np.copy(self.pos_estimate))

    def set_plot_colors(self, color):
        self.train_color = color[0]
        self.inference_color = color[1]

    def plot_path(self, ax, plot_model_trajectory=True):
        x, y, z = self.pos
        ax.plot([x], [y], [z], color=self.train_color, marker='.', label='', markersize=C.MARKERSIZE)

        if len(self.H_pos) > 0:
            s_hist = np.vstack(self.H_pos)
            ax.plot(s_hist[:, 0], s_hist[:, 1], s_hist[:, 2], color=self.train_color, linestyle=':', label='',
                    markersize=C.MARKERSIZE, alpha=C.PATH_TRANSPARENCY)

        if plot_model_trajectory:  # If in inference:
            x, y, z = self.pos_estimate

            ax.plot([x], [y], [z], color=self.inference_color, marker='.', label='',
                    markersize=C.MARKERSIZE)

            if len(self.H_pos_estimate) > 0:
                s_hist = np.vstack(self.H_pos_estimate)
                ax.plot(s_hist[:, 0], s_hist[:, 1], s_hist[:, 2], color=self.inference_color, linestyle=':',
                        label='', markersize=C.MARKERSIZE, alpha=C.PATH_TRANSPARENCY)
            # self.plot_drone_variance(ax)

    def plot_drone_variance(self, ax):
        x, y, z = self.pos_estimate
        std_x, std_y, std_z = self.pos_variance
        if self.pos_variance[0] > 0:
            # Plot the prediction uncertainty (variance)
            var_x = 3 * std_x * self.var_x_factor
            var_y = 3 * std_y * self.var_y_factor
            var_z = 3 * std_z * self.var_z_factor
            # Plot the surface
            ax.plot_surface(var_x + x, var_y + y, var_z + z, color=self.inference_color, alpha=C.DRONE_VARIANCE_TRANSPARENCY)


# Returns a value that is max(abs(max_a), abs(a+b))
def clamp_add(a, b, max_a):
    if (a + b) > max_a:
        return max_a
    if (a + b) < -1.0 * max_a:
        return -1.0 * max_a
    return a + b
import numpy as np
# from simple_pid import
import simple_pid
from . import constants as C


class Drone():
    def __init__(self):
        # State Vectors
        self.pos = np.zeros((3))  # ACTUAL Location
        self.vel = np.zeros((3))  # Velocity based on self.pos
        self.acc = np.zeros((3))  # Acceleration based on self.pos

        # Target Vector - Tracking Next Target Position
        self.target = np.zeros((3))  # Actual position target used for setpoints
        self.saved_target = np.zeros((3))  # Planned target - next waypoint. 'Goal'

        # Controllers
        self.PID_X = None
        self.PID_Y = None
        self.PID_Z = None

        self.PID_X_estimate = None
        self.PID_Y_estimate = None
        self.PID_Z_estimate = None

    def set_target(self, t):
        self.target = np.copy(t)
        self.init_PIDs()

    def init_PIDs(self):
        # Assuming we call this everytime we update the targets
        x, y, z = self.target
        self.PID_X = simple_pid.PID(C.PID_P, C.PID_I, C.PID_D, setpoint=x, sample_time=None)
        self.PID_Y = simple_pid.PID(C.PID_P, C.PID_I, C.PID_D, setpoint=y, sample_time=None)
        self.PID_Z = simple_pid.PID(C.PID_P, C.PID_I, C.PID_D, setpoint=z, sample_time=None)

        self.PID_Y_estimate = simple_pid.PID(C.PID_P, C.PID_I, C.PID_D, setpoint=y, sample_time=None)
        self.PID_X_estimate = simple_pid.PID(C.PID_P, C.PID_I, C.PID_D, setpoint=x, sample_time=None)
        self.PID_Z_estimate = simple_pid.PID(C.PID_P, C.PID_I, C.PID_D, setpoint=z, sample_time=None)

    def update_state_from_pos(self, pos):
        # Update output limits manually so that we wouldn't clamp the last integral and the last output value
        self.PID_X.output_limits = (-1.0 * C.MAX_ACC - self.acc[0], C.MAX_ACC - self.acc[0])
        self.PID_Y.output_limits = (-1.0 * C.MAX_ACC - self.acc[1], C.MAX_ACC - self.acc[1])
        self.PID_Z.output_limits = (-1.0 * C.MAX_ACC - self.acc[2], C.MAX_ACC - self.acc[2])

        # Changes in acc are the outputs of the PID controllers
        dAcc_x = self.PID_X(pos[0], dt=C.DT)
        dAcc_y = self.PID_Y(pos[1], dt=C.DT)
        dAcc_z = self.PID_Z(pos[2], dt=C.DT)

        # Update acc's by clamp adding differences in acceleration (previously clamped by setting output_limits)
        self.acc = np.asarray([self.acc[0] + dAcc_x, self.acc[1] + dAcc_y, self.acc[2] + dAcc_z])

        # Update velocities by clamp adding the velocity contributions obtained from accelerating DT 'seconds'.
        n_vel_x = clamp_add(self.vel[0], self.acc[0] * C.DT, C.MAX_VEL)
        n_vel_y = clamp_add(self.vel[1], self.acc[1] * C.DT, C.MAX_VEL)
        n_vel_z = clamp_add(self.vel[2], self.acc[2] * C.DT, C.MAX_VEL)

        self.vel = np.asarray([n_vel_x, n_vel_y, n_vel_z])
        # self.vel /= np.linalg.norm(self.vel)

        self.pos += self.vel

    def has_reached_target(self, epsilon):
        # return true if distance to target is within epsilon
        return abs(np.linalg.norm(self.pos - self.target)) < epsilon


# Returns a value that is max(abs(max_a), abs(a+b))
def clamp_add(a, b, max_a):
    if (a + b) > max_a:
        return max_a
    if (a + b) < -1.0 * max_a:
        return -1.0 * max_a
    return a + b

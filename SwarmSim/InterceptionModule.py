from . import constants as C

class InterceptionModule():
    def __init__(self, swarm_friendly, drone_adversarial):
        self.s_friendly = swarm_friendly
        self.d_adversarial = drone_adversarial

        self.adversarial_pos_window = []

    def observe_adversarial_uav(self, use_model=False):
        if use_model:
            #  IF using model need to check if in training
            pass
        else:
            self.adversarial_pos_window.append(self.d_adversarial.pos)

        if len(self.adversarial_pos_window) > C.INTERCEPTION_WIN_SIZE:
            self.adversarial_pos_window.pop(0)

    def update_friendly_swarm_trajectory(self):
        #if window full
        #   Update PID targets (w.r.t to the swarm center and initial positions)
        pass

    # Create collision check between swarm and drone
    # Implement a binary-based interception statistics

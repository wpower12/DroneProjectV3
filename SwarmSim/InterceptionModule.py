from . import constants as C

class InterceptionModule():
    def __init__(self, swarm_friendly, swarm_adversarial):
        self.s_friendly = swarm_friendly
        self.s_adversarial = swarm_adversarial

        # Assume just 1 drone in adversarial swarm
        self.d_adversarial = swarm_adversarial.drones[0]
        self.adversarial_pos_window = []

    def observe_adversarial_uav(self, use_model=False):
        if use_model:
            if self.s_adversarial.training:
                self.adversarial_pos_window.append(self.d_adversarial.pos)
            else:
                self.adversarial_pos_window.append(self.d_adversarial.pos_estimate)
        else:
            self.adversarial_pos_window.append(self.d_adversarial.pos)

        if len(self.adversarial_pos_window) > C.INTERCEPTION_WIN_SIZE:
            self.adversarial_pos_window.pop(0)
        print(self.adversarial_pos_window)

    def update_friendly_swarm_trajectory(self):
        if len(self.adversarial_pos_window) == C.INTERCEPTION_WIN_SIZE:
            self.update_friendly_targets()


    # Create collision check between swarm and drone
    # Implement a binary-based interception statistics

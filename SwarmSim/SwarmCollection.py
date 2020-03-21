import numpy as np
import time
from . import Swarm as S
from . import MultiSwarmAnimator as A
from . import constants as C

class SwarmCollection():
    def __init__(self, rnd_state):
        self.swarm_id = 0
        self.swarms = []
        self.lower_limit = None
        self.upper_limit = None
        self.animator = None
        self.prng = rnd_state

    def clearAll(self):
        self.swarms.clear()

    def add_swarm(self, swarmOptions):
        swarm = S.Swarm(swarmOptions)
        self.swarm_id += 1
        swarm.set_swarm_id(self.swarm_id)
        swarm.generate_model(prng=self.prng)

        self.update_lower_limits(start_pt=swarmOptions[3])
        self.update_upper_limits(end_pt=swarmOptions[4])

        self.swarms.append(swarm)

    def add_swarms(self, swarmsOptions):
        for swarm_option in swarmsOptions:
            self.add_swarm(swarm_option)

    def update_lower_limits(self, start_pt):
        if self.lower_limit is None:
            self.lower_limit = np.asarray(start_pt)
        else:
            self.lower_limit = np.minimum(self.lower_limit, start_pt)

    def update_upper_limits(self, end_pt):
        if self.upper_limit is None:
            self.upper_limit = np.asarray(end_pt)
        else:
            self.upper_limit = np.maximum(self.upper_limit, end_pt)

    def generate_animator(self, figname):
        if C.ANIMATE:
            self.animator = A.MultiSwarmAnimator(self.lower_limit, self.upper_limit, figname)

    def animate_current_state(self):
        if C.ANIMATE and self.animator is not None:
            self.animator.plot_swarms(self.swarms)


from . import SwarmCollection as SC
from . import Wind  as W
from . import MultiSwarmAnimator as A
from . import constants as C

import numpy as np
import time

class MultiSwarmSim():
    def __init__(self, swarms_options, rnd_seed, figname):
        # Assuming swarms holds a list of swarms:
        # [[num_drones, type, plot_color, inital_position, target],..,[...]]
        self.prng = np.random.RandomState(rnd_seed)
        self.swarms = SC.SwarmCollection(self.prng)

        self.swarms.add_swarms(swarms_options)
        self.swarms.generate_animator(figname)

        self.wind = W.Wind(self.prng)

    def set_seed(self, n):
        self.wind.set_seed(n)

    def tick(self):
        self.swarms.animate_current_state()
        self.wind.sample_wind()
        for s in self.swarms.swarms:
            s.tick(self.wind)

    def start_inference(self, use_model=True):
        for s in self.swarms.swarms:
            s.training = False
            s.use_model = use_model


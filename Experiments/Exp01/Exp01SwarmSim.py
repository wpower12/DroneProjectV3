import numpy as np

from SwarmSim import Wind  as W
from SwarmSim import Swarm as S
from SwarmSim import Interception

from . import constantsExp01 as C

class Exp01SwarmSim():
    def __init__(self, options_blue, options_red, rnd_seed, animator=None):

        self.prng = np.random.RandomState(rnd_seed)
        self.wind = W.Wind(self.prng)

        self.swarm_blue = S.Swarm(options_blue)
        self.swarm_blue.set_swarm_id(1)
        self.swarm_red  = S.Swarm(options_red)
        self.swarm_red.set_swarm_id(0)

        self.intercepting = False
        self.intercept_timer = C.INTERCEPT_UPDATE_INTERVAL

        if animator:
            self.anm = animator

    def set_seed(self, n):
        self.wind.set_seed(n)

    def tick(self):
        self.wind.sample_wind()
        wind_dev = self.wind.get_wind_vec() * C.DT

        if self.wind.gusting:
            self.swarm_blue.move_swarm(wind_dev)
            self.swarm_red.move_swarm(wind_dev)

        self.swarm_blue.tick(wind_dev)
        self.swarm_red.tick(wind_dev)

        if self.intercepting:
            if self.intercept_timer >= C.INTERCEPT_UPDATE_INTERVAL:
                self.update_blue_target()
                self.intercept_timer = 0
            else:
                self.intercept_timer += 0

        if self.anm:
            self.anm.plot(self)

    def start_inference(self, use_model=True):
        for s in self.swarms.swarms:
            s.training = False
            s.use_model = use_model

    def attempt_intercept_naieve(self):
        self.intercepting = True

    def update_blue_target(self):
        # Get window of blue swarms position
        blue_center_pos = self.get_window_blue() # (W, 3) - One center position for each t in Window

        # Get the red position window
        red_p = self.get_window_red()  # (W, 3) - Same as blue_centers, for each Window, we have a 3 vec center pos

        # Pass to intercept method.
        new_center_target = Interception.interceptNaive(red_p, blue_center_pos, C.DT, C.V_BLUE_NOM, C.INTERCEPT_SCALE)

        # Update targets for blue swarm
        self.swarm_blue.set_swarm_target_from_new_center(new_center_target)

    def get_window_blue(self):
        window = np.zeros((C.WINDOW_SIZE, len(self.swarm_blue.drones), 3))
        for t in range(C.WINDOW_SIZE):
            ps = []
            for i, d in enumerate(self.swarm_blue.drones):
                window[t][i] = d.H_pos[t]
        return np.mean(window, axis=1)

    def get_window_red(self):
        window = np.zeros((C.WINDOW_SIZE, 3))
        for t in range(C.WINDOW_SIZE):
            window[t] = self.swarm_red.drones[0].H_pos[t]
        return window

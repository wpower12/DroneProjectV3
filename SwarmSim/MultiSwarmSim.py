from . import SwarmCollection as SC
from . import Wind  as W
from . import constants as C

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import time

class MultiSwarmSim():
    def __init__(self, swarms_options, rnd_seed, figname):
        # Assuming swarms holds a list of swarms:
        # [[num_drones, type, plot_color, inital_position, target],..,[...]]
        self.prng = np.random.RandomState(rnd_seed)
        self.swarms = SC.SwarmCollection(self.prng)

        self.swarms.add_swarms(swarms_options)

        self.wind = W.Wind(self.prng)
        self.ax = self.init_figure(figname)


    def set_seed(self, n):
        self.wind.set_seed(n)

    def tick(self):
        self.swarms.plot_all_swarms(self.ax)
        self.wind.sample_wind()

        wind_dev = self.wind.get_wind_vec() * C.DT
        if self.wind.gusting:
            self.swarms.move_swarms(wind_dev)

        for s in self.swarms.swarms:
            s.tick(self.wind)

    def start_inference(self, use_model=True):
        for s in self.swarms.swarms:
            s.training = False
            s.use_model = use_model

    def init_figure(self, figname):
        plt.ion()
        fig = plt.figure(num=figname)
        ax = fig.add_subplot(111, projection='3d')
        fig.set_figheight(11)
        fig.set_figwidth(12)
        return ax


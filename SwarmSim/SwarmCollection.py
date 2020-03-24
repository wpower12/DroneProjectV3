import numpy as np
import time
from . import Swarm as S
from . import constants as C

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

class SwarmCollection():
    def __init__(self, rnd_state):
        self.swarm_id = 0
        self.swarms = []
        self.lower_limit = None
        self.upper_limit = None
        self.prng = rnd_state

    def clearAll(self):
        self.swarms.clear()

    def add_swarm(self, swarmOptions):
        swarm = S.Swarm(swarmOptions)
        self.swarm_id += 1
        swarm.set_swarm_id(self.swarm_id)
        swarm.generate_model(prng=self.prng)

        self.update_lower_limits(start_pt=swarmOptions[3])
        self.update_lower_limits(start_pt=swarmOptions[4])
        self.update_upper_limits(end_pt=swarmOptions[3])
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

    def move_swarms(self, shift_vec, weights=None):
        if weights is None:
            weights = np.ones((len(self.swarms), 3))
        for index, s in enumerate(self.swarms):
            s.move_swarm([shift_vec[0]*weights[index][0], shift_vec[1]*weights[index][1], shift_vec[2] * weights[index][2]])

    def plot_all_swarms(self, ax):
        plt.cla()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        for s in self.swarms:
            s.plot_swarm(ax)
            s.plot_swarm_variance(ax)

        self.add_legend_to_plot(ax)
        self.set_plot_limits(ax)
        # self.ax.view_init(5, -70)
        plt.gcf().canvas.draw_idle()
        plt.gcf().canvas.start_event_loop(0.001)
        # plt.savefig('/home/daniel/Documents/Drone Project/Paper Feb/Experiment 3 Visuals/Pics/frame_' + str(self.frame_num) + '.pdf')

    def add_legend_to_plot(self, ax):
        leg = []
        custom_lines = []
        for i, s in enumerate(self.swarms):
            custom_lines.append(Line2D([0], [0], color=s.color[0], linestyle=':'))
            custom_lines.append(Line2D([0], [0], color=s.color[1], linestyle=':'))
            leg.append('Swarm #' + str(i) + ' (ground truth)')
            leg.append('Swarm #' + str(i) + ' (inference)')

        ax.legend(custom_lines, leg, loc='lower right', bbox_to_anchor=(0.37, .69), prop={'size': 9})

    def set_plot_limits(self, ax):
        plt.xlim(self.lower_limit[0], self.upper_limit[0])
        plt.ylim(self.lower_limit[1], self.upper_limit[1])
        ax.set_zlim(self.lower_limit[2], self.upper_limit[2])

    def train(self):
        for s in self.swarms:
            s.train()
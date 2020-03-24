import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d


""" Expo 01 Animator

Tasked with animating the drones as they attempt to collide. Would like to show each drone, with
appropriate colors, and maybe, as a stretch goal, to show the current intersection estimate. 

"""

class Exp01Animator():
    def __init__(self, fig_title):
        plt.ion()
        self.fig_title = fig_title
        self.fig = plt.figure(num=self.fig_title)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.lower = [0, 0, 0]
        self.upper = [14, 14, 14]
        self.fig.set_figheight(11)
        self.fig.set_figwidth(12)

        self.frame_num = 0

        self.markersize = 4
        self.alpha = .3

        u = np.linspace(0, 2 * np.pi, 20) #100)
        v = np.linspace(0, np.pi, 20) #100)
        self.var_x_factor = np.outer(np.cos(u), np.sin(v))
        self.var_y_factor = np.outer(np.sin(u), np.sin(v))
        self.var_z_factor = np.outer(np.ones(np.size(u)), np.cos(v))

        self.swarm_mean = np.zeros(3)
        self.swarm_xyz_upper = np.zeros(2)
        self.swarm_xyz_lower = np.zeros(2)

        self.xyz_lower_limit = np.zeros(3)
        self.xyz_upper_limit = np.zeros(3)

        self.xyz_lower_limit[0] = np.amin(self.var_x_factor)
        self.xyz_lower_limit[0] = np.amin(self.var_y_factor)
        self.xyz_lower_limit[0] = np.amin(self.var_z_factor)

        self.xyz_upper_limit[1] = np.amax(self.var_x_factor)
        self.xyz_upper_limit[1] = np.amax(self.var_y_factor)
        self.xyz_upper_limit[1] = np.amax(self.var_z_factor)

        self.began_calculating_swarm_variance = False

    def plot(self, sim):
        self.plot_targets(sim.swarm_blue)
        self.plot_swarms([sim.swarm_blue, sim.swarm_red])

    def plot_swarms(self, swarms):
        self.frame_num +=1
        plt.cla()
        leg = []
        custom_lines = []
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.set_zticklabels([])
        for i, s in enumerate(swarms):
            self.train_color = s.color[0]
            self.inference_color = s.color[1]
            custom_lines.append(Line2D([0], [0], color=s.color[0], linestyle=':'))
            custom_lines.append(Line2D([0], [0], color=s.color[1], linestyle=':'))
            # leg.append('Swarm #' + str(i) + ' (ground truth)')
            # leg.append('Swarm #' + str(i) + ' (inference)')
            for d in s.drones:
                self.plot_drone(d, s.training)
            if not s.training:
                self.plotSwarmVariance(s)


        self.ax.legend(custom_lines, leg, loc='lower right', bbox_to_anchor=(0.37, .69), prop={'size': 9})
        plt.xlim(self.lower[0], self.upper[0])
        plt.ylim(self.lower[1], self.upper[1])
        self.ax.set_zlim(self.lower[2], self.upper[2])
        #self.ax.view_init(5, -70)
        plt.gcf().canvas.draw_idle()
        plt.gcf().canvas.start_event_loop(0.001)


    def plot_drone(self, d, in_training=True):
        x, y, z = d.pos
        self.ax.plot([x], [y], [z], color=self.train_color, marker='.', label='', markersize=self.markersize)

        if len(d.H_pos) > 0:
            s_hist = np.vstack(d.H_pos)
            self.ax.plot(s_hist[:, 0], s_hist[:, 1], s_hist[:, 2], color=self.train_color, linestyle=':', label='',
                         markersize=self.markersize, alpha=self.alpha)

        if not in_training:  # If in inference:
            x, y, z = d.pos_estimate_animate
            std_x, std_y, std_z = d.pos_variance
            self.ax.plot([x], [y], [z], color=self.inference_color, marker='.', label='', markersize=self.markersize)

            # if d.pos_variance[0] > 0:
            #    # Plot the prediction uncertainty (variance)
            #    var_x = 3 * std_x * self.var_x_factor
            #    var_y = 3 * std_y * self.var_y_factor
            #    var_z = 3 * std_z * self.var_z_factor
            #    # Plot the surface
            #    self.ax.plot_surface(var_x + x, var_y + y, var_z + z, color=self.inference_color, alpha=.2)

            if len(d.H_pos_estimate) > 0:
                s_hist = np.vstack(d.H_pos_estimate)
                self.ax.plot(s_hist[:, 0], s_hist[:, 1], s_hist[:, 2], color=self.inference_color, linestyle=':',
                             label='', markersize=self.markersize, alpha=self.alpha)


    def plot_targets(self, swarm):
        x, y, z = [], [], []

        for d in swarm.drones:
            x_d, y_d, z_d = d.target
            x.append(x_d)
            y.append(y_d)
            z.append(z_d)
        self.ax.plot(np.asarray(x), np.asarray(y), np.asarray(z), color='r', marker='+', markersize=self.markersize)

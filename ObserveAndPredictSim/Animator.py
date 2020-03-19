import matplotlib.pyplot as plt


class Animator():
    def __init__(self, start, destination,  fig_title):
        plt.ion()
        self.fig_title = fig_title
        self.fig = plt.figure(num=fig_title)
        self.ax = self.fig.add_subplot(111)
        self.start = start
        self.destination = destination
        self.color_str = "b."

    def plot(self, swarm):
        plt.cla()

        # Drones
        for d in swarm.drones:
            x, y, z = d.pos
            self.ax.plot([x], [y], [z], self.color_str)

        # History
        # self.color_str = "b."
        # for t in swarm.position_history:
        #     for d in t:
        #         x, y, z = d
        #         self.ax.plot([x], [y], [z], self.color_str)

        # plt.xlim(self.start[0], self.destination[0])
        # plt.ylim(self.start[1], self.destination[1])
        # self.ax.set_zlim(self.start[2], self.destination[2])
        #
        # self.ax.set_xticklabels([])
        # self.ax.set_yticklabels([])
        # self.ax.set_zticklabels([])
        plt.show()
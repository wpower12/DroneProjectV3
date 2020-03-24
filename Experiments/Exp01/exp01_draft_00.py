# from . import Exp01Animator as A
# from . import Exp01SwarmSim as S
# from SwarmSim.ExpHelper import *
# from . import constantsExp01 as C
import numpy as np
import random
import warnings

import Experiments.Exp01.Exp01SwarmSim as S
import Experiments.Exp01.Exp01Animator as A
import Experiments.Exp01.constantsExp01 as C

""" Experiment 01 - Single Drone Intercepted by Swarm, Full Observations

"""

RND_SEED = 10

swarm_blue = [4, 'planar', ['b', 'b'], [0, 0, 0], [12, 12, 12], None]
swarm_red  = [1, 'planar',  ['r', 'xkcd:orange'], [14, 14, 14], [1, 1, 1], None]

anm = A.Exp01Animator("testing exp1 intersection")
sim = S.Exp01SwarmSim(swarm_blue, swarm_red, RND_SEED,  anm)

for t in range(C.NUM_STEPS_PRE):
    sim.tick()

sim.attempt_intercept_naieve()

for t in range(C.NUM_STEPS_POST):
    sim.tick()

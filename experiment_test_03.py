from SwarmSim.MultiSwarmSim import *
from SwarmSim.ExpHelper import *
from SwarmSim import Swarm as S
from SwarmSim import constants as C
import numpy as np
import random
import warnings

warnings.filterwarnings("ignore")

# [Number of drones, Color, StartPoint, EndPoint, UsingStructure]
# Color = [ColorInTrain, ColorInInference]
swarm1 = [4, 'planar', ['b', 'r'], [2, 0, 0], [13, 13, 13], None]
swarm2 = [4, 'planar',  ['g', 'xkcd:orange'], [12, 13, 13], [0, 0, 0], None]
multiswarm_options = [swarm1, swarm2]


for n in range(C.NUM_RUNS):
    print('run:', n + 1)

    # rnd_seed = 0 # when set to 0 we don't get the LINALG exception
    rnd_seed = random.randint(0, 10000000)
    rnd_seed=2265069 # DEAD RECONING PATH ?!?!?!
    print('rnd_seed:', rnd_seed)
    print('\n\n')


    # =========================================================================
    '''
    # baseline
    np.random.seed = rnd_seed
    sim = MultiSwarmSim(multiswarm_options, rnd_seed, 'Ground truth')
    sim.set_seed(rnd_seed)
    for i in range(C.NUM_TRAINING_STEPS + C.NUM_INFERENCE_STEPS):
        sim.tick()

    print('Ground truth trajectories: generated!')
    print('\n\n')
    '''
    # =========================================================================
    '''
    # without Model
    np.random.seed = rnd_seed
    sim = MultiSwarmSim(multiswarm_options, rnd_seed, C.NUM_TRAINING_STEPS, C.NUM_INFERENCE_STEPS, None, 'Dead reckoning', C.ANIMATE)
    sim.set_seed(rnd_seed)
    for i in range(C.NUM_TRAINING_STEPS):
        sim.tick()

    # starting inference with FALSE means we DONT use the model
    # only dead reckoning
    sim.start_inference(False)
    for i in range(C.NUM_INFERENCE_STEPS):
        sim.tick()

    # =========================================================================

    # with Model
    np.random.seed = rnd_seed
    sim = MultiSwarmSim(multiswarm_options, rnd_seed, C.NUM_TRAINING_STEPS, C.NUM_INFERENCE_STEPS, False, 'Unstructured regressor', C.ANIMATE)
    sim.set_seed(rnd_seed)
    for i in range(C.NUM_TRAINING_STEPS):
        sim.tick()

    # starting inference with TRUE means we use the model
    sim.start_inference(True)
    for i in range(C.NUM_INFERENCE_STEPS):
        sim.tick()

    print('\n\n')
    
    # =========================================================================
    '''
    # with Model
    multiswarm_options[0][-1] = multiswarm_options[1][-1] = True
    np.random.seed = rnd_seed
    sim = MultiSwarmSim(multiswarm_options, rnd_seed, 'Temporal GCRF')
    sim.set_seed(rnd_seed)
    sim.init_intercept_module()
    for i in range(C.NUM_TRAINING_STEPS+C.NUM_INFERENCE_STEPS):
        sim.tick()


print('Done!')

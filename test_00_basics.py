from ObserveAndPredictSim import Swarm, Observer, Animator

NUM_TIMESTEPS = 100
START = [0,0,0]
END = [20,20,20]
NUM_DRONES = 27

swarm = Swarm.Swarm(NUM_DRONES)
obs = Observer.Observer()
anm = Animator.Animator(START, END, "test")
swarm.set_swarm_target_relative(END)

for i in range(NUM_TIMESTEPS):
    swarm.tick()
    obs.observe(swarm)
    anm.plot(swarm)
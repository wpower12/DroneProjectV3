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

        # Colision avoidance vars
        self.did_colision_avoidance = False
        self.post_colision_avoidance_timer = C.WINDOW_SIZE
        self.saved_last_swarm_pos_estimates = []
        self.delta_vec = None

    def clearAll(self):
        self.swarms.clear()

    def addSwarm(self, swarmOptions):
        swarm = S.Swarm(swarmOptions)
        self.swarm_id += 1
        swarm.set_swarm_id(self.swarm_id)
        swarm.generate_model(prng=self.prng)

        self.update_lower_limits(start_pt=swarmOptions[3])
        self.update_upper_limits(end_pt=swarmOptions[4])

        self.swarms.append(swarm)

    def addSwarms(self, swarmsOptions):
        for swarm_option in swarmsOptions:
            self.addSwarm(swarm_option)

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


# Colision Avoidance Mechanism ########################
#######################################################
    def resolvePotentialCollisions(self):
        if self.swarmsIntersect():
            for s in self.swarms:
                swarm_pos_estimates = np.zeros((s.N, 3), dtype=float)
                for i, d in enumerate(s.drones):
                    swarm_pos_estimates[i, :] = np.copy(d.pos_estimate)

                self.saved_last_swarm_pos_estimates.append(swarm_pos_estimates)
            while self.swarmsIntersect():
                deflection_vectors = self.getDeflectionVectors()
                self.singleStepCollisionAvoidance(deflection_vectors)
                self.animator.plot_swarms(self.swarms)
            self.did_colision_avoidance = True

        if self.delta_vec is None and self.did_colision_avoidance:
            self.delta_vec = []
            for si, s in enumerate(self.swarms):
                swarm_delta_vects = np.zeros((s.N, 3))
                for di, d in enumerate(s.drones):
                    swarm_delta_vects[di, :] = d.pos_estimate_animate - self.saved_last_swarm_pos_estimates[si][di, :]
                self.delta_vec.append(swarm_delta_vects)
        if self.did_colision_avoidance:
            for si, s in enumerate(self.swarms):
                for di, d in enumerate(s.drones):
                    print("Usao za di ", di)
                    d.pos_estimate_animate = d.pos_estimate + self.delta_vec[si][di, :]
        else:
            for s in self.swarms:
                for d in s.drones:
                    d.pos_estimate_animate = d.pos_estimate
        for s in self.swarms:
            if not s.training:
                for d in s.drones:
                    d.H_pos_estimate.append(d.pos_estimate_animate)

        if self.swarms[0].timestep == (self.swarms[0].num_inference_steps + self.swarms[0].num_training_steps - 1):
            time.sleep(100)

    def swarmsIntersect(self):
        if self.swarms[0].training:
            return False

        std_1 = self.swarms[0].swarm_variance
        std_2 = self.swarms[1].swarm_variance

        if std_1[0] > 0 and std_2[0] > 0:

            std_1 += C.DT
            std_2 += C.DT

            dist12 = np.linalg.norm(self.swarms[0].swarm_mean - self.swarms[1].swarm_mean)

            if np.any(std_1+std_2 > dist12):
                print('COLISION!')
                return True

        return False

    def getDeflectionVectors(self):
        deflection_vectors = np.zeros((2,3), dtype=float)
        if self.swarms[0].swarm_mean[2] > self.swarms[1].swarm_mean[2]:
            # 0th swarm goes up
            deflection_vectors[0,:] = np.asarray([0.0,0.0,1.0])
            deflection_vectors[1,:] = -np.asarray([0.0,0.0,1.0])
        else:
            # 0th swarms goes up
            deflection_vectors[0, :] = -np.asarray([0.0, 0.0, 1.0])
            deflection_vectors[1, :] = np.asarray([0.0, 0.0, 1.0])

        return deflection_vectors

    def singleStepCollisionAvoidance(self, deflectionVectors):
        deflectionVectors[0, :] +=  np.asarray([1.0, 1.0, 1.0])
        deflectionVectors[1, :] += -np.asarray([1.0, 1.0, 1.0])

        deflectionVectors[0, :] /= np.linalg.norm(deflectionVectors[0, :])
        deflectionVectors[1, :] /= np.linalg.norm(deflectionVectors[1, :])

        deflectionVectors *= C.DT

        for i, s in enumerate(self.swarms):
            for d in s.drones:
                #d.update_state_from_pos_estimate(d.pos_estimate_animate) # this is the first 4 lines
                d.pos_estimate_animate += deflectionVectors[i, :]
                d.H_pos_estimate.append(np.copy(d.pos_estimate_animate))


                s.updateSwarmMeanPostCollisionAvoidance()
                

#######################################################

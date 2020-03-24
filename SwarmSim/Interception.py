import numpy as np

def interceptNaive(red, blue, dt, V_BLUE_NOM, INTERCEPT_SCALE):
    print("running intercept method: naieve")
    print(red, blue)

    v_red = get_v_from_W(red, dt)

    # Now we get the normalized v_red
    norm_v_red = v_red / np.linalg.norm(v_red)

    # print("norm: ", norm_v_red)

    # Use distnace between two swarms at most recent time step to
    # define a 'time to hit if we just stayed here'. This is a rough
    # estimate of the time scale needed to 'close' the distance
    # between the swarms
    t_hit_naieve = np.linalg.norm(red[0] - blue[0]) / V_BLUE_NOM

    # print("t_hit ", t_hit_naieve)

    # Our 'guess' at an intercept is then the point along norm_v_red
    # that is t_hit_naive away
    pred_coll_target = red[0] + t_hit_naieve * norm_v_red * INTERCEPT_SCALE

    return pred_coll_target


def get_v_from_W(swarm, sample_dt):
    vel_diffs = []
    for i in range(len(swarm) - 1):
        vel_diffs.append((swarm[i + 1] - swarm[i]) / sample_dt)
    return np.asarray(vel_diffs).sum(axis=0) / (len(vel_diffs))


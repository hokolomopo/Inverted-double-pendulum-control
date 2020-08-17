import numpy as np
import time
import math 

from agent import Agent
from policy import RandomPolicy

def get_dataset(n_trajectories=100, len_trajectories=1000, policy=RandomPolicy(), list_of_traj=False, verbose=True, env=None):
    """ Generate a dataset for FQI."""

    X = []
    X_next = []
    Y = []

    start_time = time.time()

    for j in range(n_trajectories):
        if verbose and j % (math.ceil(n_trajectories/10)) == 0:
            remaining_iterations = n_trajectories - j
            elapsed_time = time.time() - start_time
            remaining_time = 0
            if j > 0:
                remaining_time = elapsed_time / j * remaining_iterations
            print("Dataset generated at {}%, elapsed time {:.0f}s, remaining time {:.0f}s".format(int(j/n_trajectories * 100), elapsed_time, remaining_time))


        traj, rewards, next = Agent.generate_trajectory(len_trajectories, policy=RandomPolicy(), stop_at_terminal=False, env=env)

        if list_of_traj:
            X.append(traj)
            X_next.append(next)
            Y.append(rewards)
        else:
            X.extend(traj)
            X_next.extend(next)
            Y.extend(rewards)

        
    return [np.array(X), np.array(Y), np.array(X_next)]

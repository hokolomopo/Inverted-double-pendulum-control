import time
import numpy as np
import dataset_util
from util import *

from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt

import gym  # open ai gym
import pybulletgym  # register PyBullet enviroments with open ai gym

SEED = 1565

from policy import OptimalPolicyDiscrete, RandomPolicy
from agent import Agent


def FQI(possible_actions, iterations, verbose=True, gamma=0.99, env=None):
    """ FQI algorithm. Take as input a Learning set (X, Y, and X_next) and the name of the model to use"""

    model = ExtraTreesRegressor(50)
    # Y_0 = Y

    start_time = time.time()

    alive_times = []
    rewards_means = []

    policy = RandomPolicy()
    X, Y, X_next = dataset_util.get_dataset(100, 200, list_of_traj=False, env=env, policy=policy, verbose=False)

    for j in range(iterations):

        Y_0 = Y

        model.fit(X, Y)

        # Update Y
        Y = []
        for i, x_next in enumerate(X_next):
            to_predict = np.array(list(map(lambda u: np.concatenate(([u], x_next)), possible_actions)))

            max_prediction = max(model.predict(to_predict))

            Y.append(Y_0[i] + gamma * max_prediction)
        
        j = j + 1

        policy = OptimalPolicyDiscrete(possible_actions, model)

        # Testing
        alive = []
        rews = []
        for k in range(128):
            init_state = env.reset()
            agent = Agent(env, init_state)
            agent.policy = policy

            done = False
            reward = 0
            decay = 1
            steps = 0
            while(done == False):
                _, r, done = agent.step()
                reward += r * decay
                decay *= 0.99
                steps += 1

            alive.append(steps)
            rews.append(reward)
        
        alive_times.append(np.mean(alive))
        rewards_means.append(np.mean(rews))

        # Printing for verbose mode
        if verbose:
            remaining_iterations = iterations - j
            elapsed_time = time.time() - start_time
            remaining_time = 0
            if j > 0:
                remaining_time = elapsed_time / j * remaining_iterations
            print("Fit {}, elapsed time {:.0f}s, remaining time {:.0f}s, alive steps = {:.1f}, reward = {:.1f}".format(j, elapsed_time, remaining_time,
                        alive_times[-1], rewards_means[-1]))

    return [model, alive_times, rewards_means]

if __name__ == "__main__":
    # env = gym.make('InvertedDoublePendulumPyBulletEnv-v0')
    env = gym.make('InvertedPendulumPyBulletEnv-v0')
    env.seed(seed=SEED)
    init_state = env.reset()  # should return a state vector if everything worked

    steps = 1000

    agent = Agent(env, init_state)

    print("#####################################################")
    print("Dataset Generation")
    print("#####################################################")


    print("#####################################################")
    print("Training")
    print("#####################################################")

    # model, alive_time, cum_rewards = FQI(X, Y, X_next, agent.get_possible_actions(), iterations=100, env=env)
    model, alive_time, cum_rewards = FQI(agent.get_possible_actions(), iterations=100, env=env)

    save_results_path("results/fqi/FQI.csv", np.array([cum_rewards, alive_time]))

    plt.plot(cum_rewards, label="Cum rewards")

    plt.plot(build_moving_average(cum_rewards, alpha=0.1), label="Average")
    plt.xlabel("Number of batches")
    plt.ylabel("Mean cumulative reward of batch")
    plt.legend()

    plt.show()


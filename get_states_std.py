import gym
import pybulletgym
import numpy as np
import random
import math

import matplotlib.pyplot as plt

# ENV_NAME = "InvertedDoublePendulumPyBulletEnv-v0"
ENV_NAME = "InvertedPendulumPyBulletEnv-v0"

MAX_EPISODES = 500
MAX_TIMESTEPS = 200

if __name__=="__main__":
    env = gym.make(ENV_NAME)
    random.seed(46454)

    states = []

    for i_episode in range(MAX_EPISODES):

        state = env.reset()

        for timesteps in range(MAX_TIMESTEPS):
            states.append(state)

            action = random.random() * 2 - 1
            state, reward, done, _ = env.step([action])

            if done:
                # print("Episode {} finished after {} timesteps".format(i_episode, timesteps+1))
                break

    env.close()
    states = np.array(states)

    print(np.mean(states, axis=0))
    print(np.std(states, axis=0))



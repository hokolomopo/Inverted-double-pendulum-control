import gym
import pybulletgym
import numpy as np
import random
import math
import time

import matplotlib.pyplot as plt

ENV_NAME = "InvertedDoublePendulumPyBulletEnv-v0"
# ENV_NAME = "InvertedPendulumPyBulletEnv-v0"

MAX_EPISODES = 100
MAX_TIMESTEPS = 200

if __name__=="__main__":
    env = gym.make(ENV_NAME)
    env.render(close=False)
    random.seed(46454)

    states = []

    for i_episode in range(MAX_EPISODES):

        state = env.reset()

        for timesteps in range(MAX_TIMESTEPS):
            states.append(state)

            action = random.random() * 2 - 1
            state, reward, done, _ = env.step([action])

            theta = math.acos(state[3]) * 180 / math.pi
            gamma = math.acos(state[6]) * 180 / math.pi

            print("Theta : {:.2f}, Gamam {:.2f}".format(theta, gamma))

            if done:
                # print("Episode {} finished after {} timesteps".format(i_episode, timesteps+1))
                break

            time.sleep(0.1)

    env.close()
    states = np.array(states)

    print(np.mean(states, axis=0))
    print(np.std(states, axis=0))
    print("Max : {}".format(np.max(states, axis=0)))
    print("Min : {}".format(np.min(states, axis=0)))



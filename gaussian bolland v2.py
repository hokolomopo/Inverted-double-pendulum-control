import gym
import pybulletgym
import numpy as np
import random
import time
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.distributions.normal import Normal
from torch.autograd import Variable
from torch import Tensor

import matplotlib.pyplot as plt

SCALE = None
STD = None

ENV_NAME = "InvertedDoublePendulumPyBulletEnv-v0"
DIMS = 9
SCALE = [ 0.00868285,  0.03400105, -0.00312787,  0.95092393, -0.01797627, -0.10439248, 0.86726532,  0.01176883,  0.12335652]
STD = [0.11101651, 0.58301397, 0.09502404, 0.07712284, 0.29911971, 1.78995357, 0.20914456, 0.45163139, 3.08248822]

ENV_NAME = "InvertedPendulumPyBulletEnv-v0"
DIMS = 5
SCALE = [-0.00207544, -0.00473823,  0.99553131,  0.00118865,  0.01364543]
STD = [0.05083671, 0.24168294, 0.00498501, 0.09429286, 0.55792934]

# ENV_NAME = "CartPole-v0"
# DIMS = 4

SEED = 464684

MAX_EPISODES = 500
BATCH_SIZE = 64
MAX_TIMESTEPS = 200

ALPHA = 0.01
GAMMA = 0.95
SIGMA = 1


class MLPPolicy(nn.Module):
    def __init__(self, input_size=5, layers=(128,), scale=None, normalize=None):
        super(MLPPolicy, self).__init__()

        if scale is None:
            self.scale = torch.zeros(1, input_size)
        else:
            self.scale = torch.tensor([scale])

        if normalize is None:
            self.normalize = torch.ones(1, input_size)
        else:
            self.normalize = torch.tensor([normalize])

        self.layers = []
        for n_neurons in layers:
            self.layers.append(nn.Linear(input_size, n_neurons))
            self.layers.append(nn.Tanh())
            input_size = n_neurons

        self.layers.append(nn.Linear(input_size, 1))

        self.net = nn.Sequential(*self.layers)

    def forward(self, x):
        return Normal(self.net((x.float() - self.scale) / self.normalize), SIGMA)

    def update_weight(self, states, actions, rewards, cum_rew, batch_size=1):

        losses = []
        optimizer.zero_grad()

        for i in range(len(states)):
            G = Variable(torch.Tensor([0]))
            for s_t, a_t, r_tt in zip(states[i][::-1], actions[i][::-1], rewards[i][::-1]):
                G = torch.Tensor([r_tt]) + GAMMA * G
                loss = (-1.0) * G * self(torch.Tensor(s_t)).log_prob(a_t)
                loss.backward()
        optimizer.step()

if __name__ == "__main__":
    policy = MLPPolicy(DIMS, scale=SCALE, normalize=STD)

    env = gym.make(ENV_NAME)
    env.seed(seed=SEED)

    optimizer = optim.Adam(policy.parameters(), lr=ALPHA)

    alive_time = []

    for i_episode in range(MAX_EPISODES):


        states = []
        actions = []
        rewards = []
        alive_time_batch = []

        for b in range(BATCH_SIZE):

            state = env.reset()
            states.append([])
            actions.append([])
            rewards.append([])

            for timesteps in range(MAX_TIMESTEPS):
                state_tensor = Tensor(state)

                with torch.no_grad():
                    action = policy(state_tensor).sample()

                states[b].append(state)
                actions[b].append(action)

                state, reward, done, _ = env.step([action.item()])
                # state, reward, done, _ = env.step(action.item())

                if reward > 0:
                    reward = 1

                if done:

                    alive_time_batch.append(timesteps+1)
                    rewards[b].append(0)
                    break

                rewards[b].append(reward)

                if timesteps == MAX_TIMESTEPS-1:
                    alive_time_batch.append(timesteps+1)


            

        policy.update_weight(states, actions, rewards, BATCH_SIZE)

        alive_time.append(np.mean(alive_time_batch))
        print("Episode {}  finished after a mean of {:.1f} timesteps and a std of {:.2f}".format(
            i_episode, np.mean(alive_time_batch), np.std(alive_time_batch)))

    env.close()

    plt.plot(alive_time)
    plt.xlabel("Number of batches")
    plt.ylabel("Iteration staying alive")
    plt.show()

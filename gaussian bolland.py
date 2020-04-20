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

# ENV_NAME = "InvertedDoublePendulumPyBulletEnv-v0"
ENV_NAME = "InvertedPendulumPyBulletEnv-v0"
DIMS = 5
SCALE = [-0.00207544, -0.00473823,  0.99553131,  0.00118865,  0.01364543]
STD = [0.05083671, 0.24168294, 0.00498501, 0.09429286, 0.55792934]


# SCALE = None
# STD = None


# ENV_NAME = "CartPole-v0"
# DIMS = 4

SEED = 464684

MAX_EPISODES = 2000
MAX_TIMESTEPS = 200

ALPHA = 0.05
GAMMA = 0.99
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
        return Normal(self.net((x - self.scale) / self.normalize), SIGMA)

    def update_weight(self, states, actions, rewards, optimizer):
        G = torch.Tensor([0])
        log_probs = []
        Gs = []
        for s_t, a_t, r_tt in zip(states[::-1], actions[::-1], rewards[::-1]):
            G = r_tt + GAMMA * G
            log_prob = self(s_t).log_prob(a_t)
            log_probs.append(log_prob)
            Gs.append(G)

        loss = torch.sum(torch.stack(log_probs)) * torch.sum(torch.stack(Gs))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    policy = MLPPolicy(DIMS, scale=SCALE, normalize=STD)

    env = gym.make(ENV_NAME)
    env.seed(seed=SEED)

    # agent.cuda()
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    optimizer = optim.Adam(policy.parameters(), lr=ALPHA)

    episodes = []
    alive_time = []

    for i_episode in range(MAX_EPISODES):

        state = env.reset()

        states = []
        actions = []
        rewards = [0]   # no reward at t = 0

        for timesteps in range(MAX_TIMESTEPS):
            state = Tensor(state)

            action = policy(state).sample()
            n = policy(state)
            log_prob = policy(state).log_prob(action)

            states.append(state)
            actions.append(action)

            state, reward, done, _ = env.step([action.item()])
            # state, reward, done, _ = env.step(action.item())

            rewards.append(reward)

            if done:
                # alive_time.append(timesteps+1)
                # if i_episode > 3:
                #     alive_time.append(np.mean([timesteps+1, alive_time[-1], alive_time[-2]]))
                # else:
                alive_time.append(timesteps+1)
                episodes.append(i_episode)
                print("Episode {} finished after {} timesteps".format(i_episode, timesteps+1))
                break

        policy.update_weight(states, actions, rewards, optimizer)

    env.close()

    plt.plot(alive_time)
    plt.show()

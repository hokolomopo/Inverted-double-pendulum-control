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
# SCALE = [-0.00207544, -0.00473823,  0.99553131,  0.00118865,  0.01364543]
# STD = [0.05083671, 0.24168294, 0.00498501, 0.09429286, 0.55792934]



SCALE = None
STD = None


# ENV_NAME = "CartPole-v0"
# DIMS = 4

SEED = 464684

MAX_EPISODES = 100
BATCH_SIZE = 64
MAX_TIMESTEPS = 200

ALPHA = 0.05
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

        # decay = torch.flip(torch.cumprod(torch.ones(len(rewards)) * GAMMA, dim=0), dims=(0,)) / GAMMA
        cum_rew_sum = torch.sum(cum_rew)


        log_probs = torch.sum(self(states).log_prob(actions))

        loss = log_probs * cum_rew_sum / batch_size

        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    policy = MLPPolicy(DIMS, scale=SCALE, normalize=STD)

    env = gym.make(ENV_NAME)
    env.seed(seed=SEED)

    optimizer = optim.Adam(policy.parameters(), lr=ALPHA)

    alive_time = []

    for i_episode in range(MAX_EPISODES):

        optimizer.zero_grad()

        states = []
        actions = []
        rewards = []
        cum_rewards = []
        alive_time_batch = []

        for b in range(BATCH_SIZE):

            state = env.reset()
            rewards_batch = []

            decay = 1

            for timesteps in range(MAX_TIMESTEPS):
                state_tensor = Tensor(state)

                with torch.no_grad():
                    action = policy(state_tensor).sample()

                states.append(state)
                actions.append(action)

                state, reward, done, _ = env.step([action.item()])
                # state, reward, done, _ = env.step(action.item())
                if done:
                    print("Episode {} batch {} finished after {} timesteps".format(i_episode, b, timesteps+1))

                    alive_time_batch.append(timesteps+1)
                    rewards.append(0)
                    break

                rewards.append(reward * decay)
                decay = decay * GAMMA
            
            rewards_batch = rewards[-timesteps-1:]
            cum_rewards.extend(np.cumsum(rewards_batch))

        policy.update_weight(torch.tensor(states), torch.tensor(actions), torch.tensor(rewards), torch.tensor(cum_rewards), BATCH_SIZE)

        alive_time.append(np.mean(alive_time_batch))

    env.close()

    plt.plot(alive_time)
    plt.show()

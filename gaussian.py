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

# ENV_NAME = "InvertedPendulumPyBulletEnv-v0"
# DIMS = 5
# SCALE = [-0.00207544, -0.00473823,  0.99553131,  0.00118865,  0.01364543]
# STD = [0.05083671, 0.24168294, 0.00498501, 0.09429286, 0.55792934]

# ENV_NAME = "CartPole-v0"
# DIMS = 4

SEED = 464684

MAX_EPISODES = 10000
BATCH_SIZE = 1
MAX_TIMESTEPS = 200

ALPHA = 0.001
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
        out = self.net((x.float() - self.scale) / self.normalize)
        return Normal(out, SIGMA)


    def update_weight(self, log_probs, rewards, batch_size=1):
        optimizer.zero_grad()

        losses = []
        for i in range(batch_size):
            R = 0
            batch_losses = []
            returns = []
            for r in rewards[i][::-1]:
                R = r + GAMMA * R
                returns.insert(0, R)
            returns = torch.tensor(returns)
            returns = returns - returns.mean()
            for log_prob, R in zip(log_probs[i], returns):
                batch_losses.append(-log_prob * R)
            loss = torch.cat(batch_losses).sum() / batch_size
            losses.append(loss)

        loss = torch.sum(torch.stack(losses))    
        loss.backward()
        optimizer.step()


def train():
    return

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
        log_probs = []
        alive_time_batch = []

        for b in range(BATCH_SIZE):

            state = env.reset()
            batch_states = []
            states.append(batch_states)
            actions.append([])
            rewards.append([])
            log_probs.append([])

            for timesteps in range(MAX_TIMESTEPS):
                state_tensor = Tensor(state)

                # with torch.no_grad():
                action = policy(state_tensor).sample()
                log_prob = policy(state_tensor).log_prob(action)
                log_probs[b].append(log_prob)

                batch_states.append(state)
                actions[b].append(action)

                state, reward, done, _ = env.step([action.item()])
                # state, reward, done, _ = env.step(action.item())

                # if reward > 0:
                #     reward = 1

                if done:

                    alive_time_batch.append(timesteps+1)
                    rewards[b].append(0)
                    break

                rewards[b].append(reward)

                if timesteps == MAX_TIMESTEPS-1:
                    alive_time_batch.append(timesteps+1)


        SCALE = np.mean(np.array(batch_states), axis = 0)
        STD = np.std(np.array(batch_states), axis = 0)
        policy.update_weight(log_probs, rewards, BATCH_SIZE)

        alive_time.append(np.mean(alive_time_batch))
        print("Episode {}  finished after a mean of {:.1f} timesteps and a std of {:.2f}".format(
            i_episode, np.mean(alive_time_batch), np.std(alive_time_batch)))

    env.close()

    plt.plot(alive_time)
    plt.xlabel("Number of batches")
    plt.ylabel("Iteration staying alive")
    plt.show()


###############################################
# One backward per traj                       #
###############################################

    # def update_weight(self, log_probs, rewards, batch_size=1):
    #     optimizer.zero_grad()

    #     for i in range(batch_size):
    #         R = 0
    #         policy_loss = []
    #         returns = []
    #         for r in rewards[i][::-1]:
    #             R = r + GAMMA * R
    #             returns.insert(0, R)
    #         returns = torch.tensor(returns)
    #         returns = (returns - returns.mean()) / (returns.std())
    #         for log_prob, R in zip(log_probs[i], returns):
    #             policy_loss.append(-log_prob * R)
    #         policy_loss = torch.cat(policy_loss).sum() / batch_size
    #         policy_loss.backward()
    #     optimizer.step()

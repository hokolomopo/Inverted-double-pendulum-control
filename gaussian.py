""" Monte-Carlo Policy Gradient """

from __future__ import print_function

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

from torch.autograd import Variable
import matplotlib.pyplot as plt

SEED = 464684

MAX_EPISODES = 1500
MAX_TIMESTEPS = 200

ALPHA = 3e-5
GAMMA = 0.99
ACTION_STEP = 2
SIGMA = 0.01

# ENV_NAME = "InvertedDoublePendulumPyBulletEnv-v0"
# ENV_NAME = "CartPole-v0"
ENV_NAME = "InvertedPendulumPyBulletEnv-v0"

class reinforce(nn.Module):

    def __init__(self):
        super(reinforce, self).__init__()

        # self.list_action = np.arange(-1, 1 + ACTION_STEP, ACTION_STEP)
        # self.list_action = [0, 1]
        self.list_action = np.array([-1, -0.5, 0, 0.5, 1])
        # self.list_action = np.array([-1, 0, 1])
        # self.list_action = np.array([-1, 1])

        # policy network
        self.fc1 = nn.Linear(5, 128)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.fc3(x)
        # x = self.softmax(x)
        return x

    def get_action(self, state, epsilon=0.5, training=True):

        # if training == True:
        state = Variable(torch.Tensor(state))
        state = torch.unsqueeze(state, 0)
        params = self.forward(state)
        mu = params[0][0].item()

        action = np.random.normal(mu, SIGMA**2)
        # if training == False:
        #     action = torch.argmax(probs)

        # action = action.data
        # action = action[0]
        
        return action
        # else:
        #     state = Variable(torch.Tensor(state))
        #     state = torch.unsqueeze(state, 0)
        #     probs = self.forward(state)
        #     mu = probs.item()
        #     return mu

    def pi(self, s, a):
        # s = Variable(torch.Tensor([s]))
        # probs = self.forward(s)
        # probs = torch.squeeze(probs, 0)
        # p = probs[np.where(self.list_action == a)]
        s = Variable(torch.Tensor([s]))
        params = self.forward(s)
        mu = params[0]
        # SIGMA = params[0][1]

        p = 1 / (math.sqrt(2*math.pi) * SIGMA)
        p *= torch.exp(-(a - mu)**2 / (2 * SIGMA**2))

        return p

    def update_weight(self, states, actions, rewards, optimizer):
        G = Variable(torch.Tensor([0]))
        # for each step of the episode t = T - 1, ..., 0
        # r_tt represents r_{t+1}
        for s_t, a_t, r_tt in zip(states[::-1], actions[::-1], rewards[::-1]):
            G = Variable(torch.Tensor([r_tt])) + GAMMA * G
            loss = (-1.0) * G * torch.log(self.pi(s_t, a_t))
            # update policy parameter \theta
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def main():

    env = gym.make(ENV_NAME)
    env.seed(seed=SEED)

    agent = reinforce()
    # agent.cuda()
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    optimizer = optim.Adam(agent.parameters(), lr=ALPHA)

    episodes = []
    alive_time = []

    for i_episode in range(MAX_EPISODES):

        state = env.reset()

        states = []
        actions = []
        rewards = [0]   # no reward at t = 0

        for timesteps in range(MAX_TIMESTEPS):

            if(i_episode != 0):
                epsilon = 1/math.sqrt(i_episode)
            action = agent.get_action(state, epsilon=0)

            states.append(state)
            actions.append(action)

            state, reward, done, _ = env.step([action])
            # state, reward, done, _ = env.step(action)

            rewards.append(reward)

            if done:
                # alive_time.append(timesteps+1)
                if i_episode > 3:
                    alive_time.append(np.mean([timesteps+1, alive_time[-1], alive_time[-2]]))
                else:
                    alive_time.append(timesteps+1)
                episodes.append(i_episode)
                print("Episode {} finished after {} timesteps".format(i_episode, timesteps+1))
                break

        agent.update_weight(states, actions, rewards, optimizer)

    env.close()

    # env = gym.make(ENV_NAME)
    # state = env.reset()
    # env.render(mode="human") 

    env = gym.make(ENV_NAME)
    # env.render(mode="human")
    duration = []

    for _ in range(50):
        # print("###############################################")
        # print("Reset Env")
        # print("###############################################")
        state = env.reset()

        for i in range(MAX_TIMESTEPS+1):
            action = agent.get_action(state, training=False)
            state, reward, done, _ = env.step([action])
            # print(action, state, i)
            # time.sleep(0.05)

            if done or i == MAX_TIMESTEPS:
                print("Lasted {} timesteps".format(i))
                duration.append(i)
                # print("\n\n\n")
                break

    env.close()
    print("Mean : {}".format(np.mean(duration)))

    plt.plot(alive_time)
    plt.show()


if __name__ == "__main__":
    random.seed(SEED)
    main()

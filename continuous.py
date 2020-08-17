import gym
import pybulletgym
import numpy as np
import random
import time
import math
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.distributions.normal import Normal
from torch.autograd import Variable
from torch import Tensor

import matplotlib.pyplot as plt

from util import *

ENV_NAME = "InvertedDoublePendulumPyBulletEnv-v0"
DIMS = 9
SCALE = [ 0.00868285,  0.03400105, -0.00312787,  0.95092393, -0.01797627, -0.10439248, 0.86726532,  0.01176883,  0.12335652]
STD = [0.11101651, 0.58301397, 0.09502404, 0.07712284, 0.29911971, 1.78995357, 0.20914456, 0.45163139, 3.08248822]

SEED = 464684

MAX_EPISODES = 500
BATCH_SIZE = 32
MAX_TIMESTEPS = 200

ALPHA = 0.001
GAMMA = 0.99
SIGMA = 1


class MLPPolicy(nn.Module):
    def __init__(self, input_size=5, layers=(128,), scale=None, normalize=None, sigma=1, gamma=0.99):
        super(MLPPolicy, self).__init__()

        self.sigma=sigma
        self.gamma=gamma

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
        return Normal(out, self.sigma)


    def update_weight(self, log_probs, rewards, optimizer, batch_size=1):
        """Update the weights of the neural network"""
        optimizer.zero_grad()

        losses = []
        for i in range(batch_size):
            R = 0
            batch_losses = []
            returns = []
            for r in rewards[i][::-1]:
                R = r + self.gamma * R
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


def train(env, params, max_episodes=1000, max_timesteps=200, dims=9, scale=None, std=None, 
                stop_if_alive_longer_than=200, stop_if_alive_longer_than_n_traj=100):
    """
    Train a model with a gaussian policy gradient

    Parameters:
    - env : the environnement
    - params : the aprameters of the training
    - max_episodes : the maximum number of batches that will be used for training
    - max_timesteps : the maximum number of timeteps for a single trajectory
    - scale : normalization for states (mean)
    - std : normalization for states (std)
    - stop_if_alive_longer_than : The training will stop if there is enough trajectories longer than this 
    - stop_if_alive_longer_than_n_traj : Number of trajectories that need to be longer than stop_if_alive_longer_than for the training to stop

    """

    policy = MLPPolicy(dims, scale=scale, normalize=std, gamma=params.gamma, sigma=params.sigma)
    optimizer = optim.Adam(policy.parameters(), lr=params.lr)

    alive_time = []
    cum_rewards = []

    solved_for_n_iter = 0

    for i_episode in range(max_episodes):

        states = []
        actions = []
        rewards = []
        log_probs = []

        # Generate a batch of trajectories
        for b in range(params.batch_size):

            state = env.reset()
            batch_states = []
            states.append(batch_states)
            actions.append([])
            rewards.append([])
            log_probs.append([])

            for timesteps in range(max_timesteps):
                state_tensor = Tensor(state)

                action = policy(state_tensor).sample()
                log_prob = policy(state_tensor).log_prob(action)
                log_probs[b].append(log_prob)

                batch_states.append(state)
                actions[b].append(action)

                state, reward, done, _ = env.step([action.item()])

                if done:
                    rewards[b].append(0)
                    break

                rewards[b].append(reward)

            # Check if we can stop the training
            if len(rewards[b]) >= stop_if_alive_longer_than:
                solved_for_n_iter +=1
            else:
                solved_for_n_iter = 0

            if solved_for_n_iter >= stop_if_alive_longer_than_n_traj:
                break

        # Check if we can stop the training
        if solved_for_n_iter >= stop_if_alive_longer_than_n_traj:
            break

        # Compute cum reward
        cum_rewards_batch = []
        alive_time_batch = []
        for r in rewards:
            cum_rewards_batch.append(get_cum_reward(r, params.gamma))
            alive_time_batch.append(len(r))

        cum = np.mean(cum_rewards_batch)

        alive_time.append(np.mean(alive_time_batch))
        cum_rewards.append(cum)

        # Update the weight of the neural network
        policy.update_weight(log_probs, rewards, optimizer, params.batch_size)

        print("Episode {}/{}  finished after a mean of {:.1f} timesteps and a std of {:.2f} and mean return of {:.2f}, min trajectory len : {}".format(
            i_episode, max_episodes, np.mean(alive_time_batch), np.std(alive_time_batch), cum, min([len(x) for x in rewards])))


    return [policy, cum_rewards, alive_time]

if __name__ == "__main__":

    env = gym.make(ENV_NAME)
    env.seed(seed=SEED)

    params = TrainingParameters(batch_size=BATCH_SIZE, n_layers=1, lr=ALPHA, gamma=GAMMA, discrete=False)

    policy, cum_rewards, alive_time = train(env, params, max_episodes=MAX_EPISODES, max_timesteps=MAX_TIMESTEPS, dims=9, scale=SCALE, std=STD,
            stop_if_alive_longer_than=128, stop_if_alive_longer_than_n_traj=150)

    env.close()

    save_results(params.get_model_name(), np.array([cum_rewards, alive_time]))

    torch.save(policy.state_dict(), "results/" + params.get_model_name())
    plt.plot(cum_rewards, label="Cum rewards")

    plt.plot(build_moving_average(cum_rewards, alpha=0.1), label="Average")
    plt.xlabel("Number of batches")
    plt.ylabel("Mean cumulative reward of batch")
    plt.legend()

    plt.show()

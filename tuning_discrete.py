import gym
import pybulletgym
import numpy as np
import torch
import random
import os
import time

import matplotlib.pyplot as plt

from discrete import train
from util import *

SEED = 415

ENV_NAME = "InvertedDoublePendulumPyBulletEnv-v0"
DIMS = 9
SCALE = [ 0.00868285,  0.03400105, -0.00312787,  0.95092393, -0.01797627, -0.10439248, 0.86726532,  0.01176883,  0.12335652]
STD = [0.11101651, 0.58301397, 0.09502404, 0.07712284, 0.29911971, 1.78995357, 0.20914456, 0.45163139, 3.08248822]

MAX_EPISODES = 10000
MAX_TIMESTEPS = 200

if __name__ == "__main__":

    para = {
        "step" : [1, 0.5],
        "alpha" : [0.001, 0.005, 0.01],
        "batch_size" : [1, 4, 16, 32]
    }

    while True:
        step = random.sample(para["step"], 1)[0]
        alpha = random.sample(para["alpha"], 1)[0]
        batch_size = random.sample(para["batch_size"], 1)[0]

        params = TrainingParameters(batch_size=batch_size, n_layers=1, lr=alpha, gamma=0.99, discrete=False, action_step=step)

        if os.path.isfile(get_data_path(params.get_model_name())):
            print("Training already done for params {} ".format(params.get_model_name()))
            continue

        env = gym.make(ENV_NAME)
        env.seed(seed=SEED)
        torch.manual_seed(SEED)


        print("########################################################################")
        print("Training {}".format(params.get_model_name()))
        print("########################################################################")

        start = time.time()
        list_actions = np.arange(-1, 1 + step, step)

        policy, cum_rewards, alive_time = train(env, params, max_episodes=int(MAX_EPISODES/batch_size), max_timesteps=MAX_TIMESTEPS, dims=9, scale=SCALE, std=STD,
                stop_if_alive_longer_than=200, stop_if_alive_longer_than_n_traj=200, list_actions=list_actions)

        env.close()

        results = np.array([cum_rewards, alive_time])

        save_results(params.get_model_name(), results)

        print("Time taken : {:.0f} seconds".format(time.time() - start))
        print("\n\n\n")
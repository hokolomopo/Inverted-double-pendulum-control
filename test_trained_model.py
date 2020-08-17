import gym  # open ai gym
import pybulletgym  # register PyBullet enviroments with open ai gym
import time
import copy
import torch 

from gaussian import MLPPolicy, SCALE, STD

SEED = 4564564

import matplotlib.pyplot as plt


if __name__ == "__main__":

    env = gym.make('InvertedDoublePendulumPyBulletEnv-v0')
    env.seed(seed=SEED)
    env.render(close=False) # call this before env.reset, if you want a window showing the environment
    state = env.reset()  # should return a state vector if everything worked

    policy = MLPPolicy(9, scale=SCALE, normalize=STD, gamma=0.99, sigma=1)
    policy.load_state_dict(torch.load("results/discrete_False_batch_32_layers_1_lr_0.001_gamma_0.99_sigma_1"))

    for _ in range(60):
        state = env.reset()  # should return a state vector if everything worked
        for i in range(400):
            action = policy(torch.Tensor(state)).sample()
            obs, rewards, done, _ = env.step([action.item()])
            if(done):
                break
            time.sleep(0.1)

        print("Survived {} steps".format(i))

    time.sleep(2)
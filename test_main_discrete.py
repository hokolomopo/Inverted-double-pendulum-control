import gym  # open ai gym
import pybulletgym  # register PyBullet enviroments with open ai gym
import time
import copy
import dataset_util
import os
import pandas as pd

from util import *

from policy import OptimalPolicyDiscrete
from agent import Agent
from FQI import FQI

SEED = 4564564

import matplotlib.pyplot as plt

class Data:
    def __init__(self, param, data):
        self.p = param
        self.it = data[0]
        self.rew = data[1]
        self.al = data[2]

if __name__ == "__main__":

    data = []
    for x in os.listdir("results/data"):
        tmp = x.split(".csv")
        x = tmp[0]
        p = TrainingParameters.from_string(x)
        d = load_results(x)
        # plt.plot(d.it * p.batch_size, build_moving_average(d.al, 0.01))
        data.append(Data(p, d))

    d = data[0]
    plt.plot(d.it, d.al, label="Steps Alive")
    plt.plot(d.it, d.rew, label="Cumuled Reward")
    # plt.plot(d.it, build_moving_average(d.al, 0.01), label="Alive Time")
    # plt.plot(d.it, build_moving_average(d.rew, 0.01), label="Reward")
    plt.xlabel("Number of batches")
    plt.legend()
    plt.savefig("results/graphs/training_discrete.eps")
    plt.show()

    print(len(data))
    plt.figure()
    for d in data:
        if d.p.lr == 0.001 and d.p.batch_size >= 16:
            plt.plot(d.it * d.p.batch_size, build_moving_average(d.al, 0.01))
    plt.show()

    # for d in data:
    #     if(d.p.batch_size > 4 and d.p.lr < 0.01):
    #         plt.plot(d.it * d.p.batch_size, build_moving_average(d.al, 0.01))
    #         print("{} mean of {:1f} max of {:.0f}".format(d.p.get_model_name(), np.mean(d.al), max(d.al)))
    # plt.show()

    dic = {
        "lr" : [],
        "step" : [],
        "batch" : [],
        "iter" : [],
        "rew" : [],
        "alive" : [],
        "mean_rew" : [],
        "mean_alive" : [],
        }
    for d in data:
        dic["lr"].append(d.p.lr)
        dic["step"].append(d.p.action_step)
        dic["batch"].append(d.p.batch_size)
        dic["iter"].append(d.it)
        dic["rew"].append(d.rew)
        dic["alive"].append(d.al)
        dic["mean_rew"].append(np.mean(d.rew))
        dic["mean_alive"].append(np.mean(d.al))

    df = pd.DataFrame(data=dic)

    para = {
        "step" : [1, 0.5],
        "lr" : [0.001, 0.005, 0.01],
        "batch" : [1, 4, 16, 32]
    }
    print(list(para))

    for item in list(para):
        values = para[item]
        for v in values:
            d = df.where(df[item] == v)
            arr = []
            print("{} = {}, mean alive time = {}, mean rew={}".format(item, v, d["mean_alive"].mean(), d["mean_rew"].mean()))


    # env = gym.make('InvertedDoublePendulumPyBulletEnv-v0')
    # env.seed(seed=SEED)
    # env.render(close=False) # call this before env.reset, if you want a window showing the environment
    # init_state = env.reset()  # should return a state vector if everything worked

    # steps = 1000

    # agent = Agent(env, init_state)

    # print("#####################################################")
    # print("Dataset Generation")
    # print("#####################################################")

    # X, Y, X_next = dataset_util.get_dataset(50, 40, list_of_traj=False)

    # print("#####################################################")
    # print("Training")
    # print("#####################################################")

    # model = FQI(X, Y, X_next, agent.get_possible_actions(), iterations=20)

    # optimal_policy = OptimalPolicyDiscrete(agent.get_possible_actions(), model)
    # agent.policy = optimal_policy

    # for i in range(steps):
    #     # obs, rewards, done, info = env.step([1])
    #     # if done:
    #     #     x = 3
    #     agent.step(verbose=True)
    #     time.sleep(0.1)

    # time.sleep(2)
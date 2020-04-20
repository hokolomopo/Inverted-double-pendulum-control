import gym  # open ai gym
import pybulletgym  # register PyBullet enviroments with open ai gym
import time
import copy
import dataset_util

from policy import OptimalPolicyDiscrete
from agent import Agent
from FQI import FQI

SEED = 4564564

if __name__ == "__main__":
    env = gym.make('InvertedDoublePendulumPyBulletEnv-v0')
    env.seed(seed=SEED)
    env.render(close=False) # call this before env.reset, if you want a window showing the environment
    init_state = env.reset()  # should return a state vector if everything worked

    steps = 1000

    agent = Agent(env, init_state)

    print("#####################################################")
    print("Dataset Generation")
    print("#####################################################")

    X, Y, X_next = dataset_util.get_dataset(50, 40, list_of_traj=False)

    print("#####################################################")
    print("Training")
    print("#####################################################")

    model = FQI(X, Y, X_next, agent.get_possible_actions(), iterations=20)

    optimal_policy = OptimalPolicyDiscrete(agent.get_possible_actions(), model)
    agent.policy = optimal_policy

    for i in range(steps):
        # obs, rewards, done, info = env.step([1])
        # if done:
        #     x = 3
        agent.step(verbose=True)
        time.sleep(0.1)

    time.sleep(2)
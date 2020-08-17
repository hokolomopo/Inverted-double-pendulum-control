from policy import AlwaysZeroPolicy
import random
import numpy as np
import gym 
import pybulletgym

from policy import RandomPolicy

class State:
    s = 0
    p = 0

    def __init__(self, p, s):
        self.s = s
        self.p = p

    def __repr__(self):
        return "State p={} s={}".format(self.p, self.s)

    @staticmethod
    def to_array(state):
        return [state.p, state.s]

class Agent:

    def __init__(self, env, state, policy=AlwaysZeroPolicy()):
        self.env = env
        self.state = state
        self.policy = policy
        self.done = False

    def set_state(self, state):
        """Set the state of the agent"""
        self.state = state

    def copy_to_state(self, state):
        """Create a copy of the agent at the given state"""
        return Agent(self.env, state=state, policy=self.policy)

    def step(self, verbose=False):
        """Update the agent following its policy"""

        if self.done == True:
            return

        action = self.policy.get_action(self)
        next_state, rewards, done, info = self.env.step([action])

        if verbose:
            print("Action taken : {}, current reward = {}, done = {}".format(action, rewards, done))
        
        self.done = done
        self.state = next_state

        return [next_state, rewards, done]

    def get_possible_actions(self, step=1):
        return np.arange(-1, 1 + step, step)

    @staticmethod
    def generate_trajectory(iterations, policy=RandomPolicy(), stop_at_terminal=True, env=None):
        """Generate a trajectory following the policy of the agent"""


        init_state = env.reset()  # should return a state vector if everything worked

        trajectory = []
        rewards = []
        x_next= []

        agent = Agent(env, init_state, policy)

        curr_state = init_state

        for _ in range(iterations):
            action = policy.get_action(agent)
            obs, reward, done, _ = env.step([action])

            # Current state
            t = [action]
            t.extend(curr_state)
            trajectory.append(t)

            # Reward
            if done: 
                reward = reward - 10
            rewards.append(reward)

            # Next state
            x_next.append(obs)
            
            if stop_at_terminal == True and done == True:
                break

        return [trajectory, rewards, x_next]
import random
from abc import ABC, abstractmethod
import numpy as np

class Policy(ABC):
    """Policy"""


    @abstractmethod
    def get_action(self, agent):
        """Return the action chose by this policy given the state of the agent"""
        pass



class RandomPolicy(Policy):
    def get_action(self, agent):
        return random.random()


class AlwaysZeroPolicy(Policy):
    def get_action(self, agent):
        return 0


class AlwaysMaxPolicy(Policy):
    def get_action(self, agent):
        return 1

class OptimalPolicyDiscrete(Policy):
    
    def __init__(self, possible_actions, model):
        self.model = model
        self.possible_actions = possible_actions
        
    def get_action(self, agent):

        to_predict = np.array([np.concatenate(([u], agent.state))
            for u in self.possible_actions])

        predicted = self.model.predict(to_predict)
        max_index = np.argmax(predicted)

        return self.possible_actions[max_index]
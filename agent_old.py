from policy import AlwaysZeroPolicy
import random
import numpy as np

class State:
    s = 0
    p = 0

    def __init__(self, p, s):
        self.s = s
        self.p = p

    @staticmethod
    def random_state():
        # Get p in [-1, 1], s in [-4, 4]
        p = (random.random() - 0.5) * 2
        s = (random.random() - 0.5) * 6
        return State(p, s)

    @staticmethod
    def random_init_state():
        # Get p in [-0.1, 0.1], s = 0
        p = (random.random() - 0.5) * 0.2
        return State(p, 0)

    def __repr__(self):
        return "State p={} s={}".format(self.p, self.s)

    @staticmethod
    def to_array(state):
        return [state.p, state.s]

class Agent:
    state = None
    domain = None
    policy = None

    def __init__(self, domain, state=None, policy=AlwaysZeroPolicy()):
        self.domain = domain
        self.state = state
        self.policy = policy

        if state == None:
            self.state = State.random_init_state()

    def set_state(self, state):
        """Set the state of the agent"""
        self.state = state

    def get_initial_state(self):
        """Get a random inital state in s=0, p ~ U(-0.1, 0.1)"""
        p = (random.random() * 2 - 1) * 0.1

        return State(p, 0)

    def copy_to_state(self, state):
        """Create a copy of the agent at the given state"""
        return Agent(self.domain, state=state, policy=self.policy)

    def update_state(self):
        """Update the agent following its policy"""
        action = self.policy.get_action(self)
        next_state = self.domain.next_state(self.state, u=action)

        self.state = next_state

    def generate_trajectory(self, iterations, stop_at_terminal=True):
        """Generate a trajectory following the policy of the agent"""

        tmp_agent = self.copy_to_state(self.state)

        trajectory = []
        for _ in range(iterations):
            
            # Get the next state
            action = self.policy.get_action(tmp_agent)
            next_state = self.domain.next_state(tmp_agent.state, u=action)
            reward = tmp_agent.domain.reward(next_state)

            trajectory.append((tmp_agent.state, action, reward, next_state))

            # Stop at terminal state if asked
            if stop_at_terminal and self.domain.is_terminal_state(next_state):
                break

            tmp_agent.set_state(next_state)

        return np.asarray(trajectory)
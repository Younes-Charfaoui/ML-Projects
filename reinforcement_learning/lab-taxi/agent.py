import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha=0.07
        self.gamma=0.8 

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        epsilon = 0.000001
        action_max = np.argmax(self.Q[state])
        probabilities = np.ones(self.nA) * epsilon / self.nA
        probabilities[action_max] = 1 - epsilon + epsilon / self.nA
        return np.random.choice(a=np.arange(self.nA), p=probabilities)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
    
        expected = self.expected_reward(next_state, 0.000001)
        self.Q[state][action] = self.Q[state][action] + self.alpha * (reward + self.gamma * expected - self.Q[state][action])

                
    def expected_reward(self, next_state, epsilon):
        probabilities = np.ones(self.nA) * epsilon / self.nA
        action_max = np.argmax(self.Q[next_state])
        probabilities[action_max] = 1 - epsilon + epsilon / self.nA
        expected = 0
        for action, prob in enumerate(probabilities):
            expected += prob * self.Q[next_state][action]
        return expected

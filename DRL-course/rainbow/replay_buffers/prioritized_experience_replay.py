from collections import namedtuple, deque

import torch


class PrioritizedReplayBuffer:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, capacity, epsilon=.001, alpha=.6, beta=.4, beta_increase=1e-2):
        """
        :param capacity: Max amount of experience saved in the structure
        :param epsilon: small value to insure all probabilities is not 0
        :param alpha: introduces some randomness and to insure we don't train the same experience and overfit
                      alpha=1 means greedy selecting the experience with highest priority
                      alpha=0 means pure uniform randomness
        :param beta: controls how much IS w affect learning
                     beta>=0, starts close to 0 and get closer and closer to 1
                     because these weights are more important in the end of learning when our q-values
                     begins to convert
        :param beta_increase: is the increase in beta for each sampling.
        """
        self.capacity = capacity
        self.memory_tree = SumTree(self.capacity)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.beta_increase = beta_increase

    def sample(self):
        pass

    def add(self, experience):
        pass

    def compute_priority(self):
        pass

    def update_memory_tree(self):
        pass


class SumTree:
    def __init__(self, capacity):

        debug = 0

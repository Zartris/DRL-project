import random
from collections import namedtuple, deque

import numpy as np
import torch


# ======================================= REPLAY BUFFERS (ER) =======================================================#

# FROM UDACITY EXAMPLE
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


# MODIFIED TO USE PRIORITY
class PrioritizedReplayBuffer:
    """Fixed-size buffer to store experience tuples. https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Dueling%20Double%20DQN%20with%20PER%20and%20fixed-q%20targets/Dueling%20Deep%20Q%20Learning%20with%20Doom%20(%2B%20double%20DQNs%20and%20Prioritized%20Experience%20Replay).ipynb"""
    absolute_error_upper = 1.  # clipped abs error

    def __init__(self, buffer_size, batch_size, seed, device, epsilon=0.001, alpha=0.6, beta=0.4, beta_increase=5e-3):
        """Initialize a ReplayBuffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            beta (float): importance-sampling, from initial value increasing to 1, often 0.4
            beta_increase (float): annealing the bias, often 1e-3
            alpha (float): [0..1] convert the importance of TD error to priority, often 0.6
            epsilon (float): small amount to avoid zero priority
        """
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.memory_tree = SumTree(buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.device = device

        self.beta = beta
        self.beta_increase = beta_increase
        self.alpha = alpha
        self.epsilon = epsilon

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory_tree)

    def is_full(self):
        return len(self.memory_tree) >= self.memory_tree.capacity

    def add(self, sample, error=None):
        if error is None:
            priority = np.amax(self.memory_tree.tree[-self.memory_tree.capacity:])
            if priority == 0:
                priority = self.absolute_error_upper
        else:
            priority = min((abs(error) + self.epsilon) ** self.alpha, self.absolute_error_upper)
        self.memory_tree.add(sample, priority)

    def sample(self):
        """
        - First, to sample a minibatch of size k the range [0, priority_total] is divided into k ranges.
        - Then a value is uniformly sampled from each range.
        - We search in the sumtree, the experience where priority score correspond to sample values are retrieved from.
        - Then, we calculate IS weights for each minibatch element.

        The difference here from the last structure is that we need to move to device
        in the method calling this function.

        so an example:
        idxs, experiences, is_weights = self.memory.sample(BATCH_SIZE)

        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        is_weights =  torch.from_numpy(is_weights).float().to(device)
        """

        minibatch = []

        idxs = np.empty((self.batch_size,), dtype=np.int32)
        is_weights = np.empty((self.batch_size,), dtype=np.float32)

        # Calculate the priority segment
        # Divide the Range[0, ptotal] into n ranges
        priority_segment = self.memory_tree.total_priority / self.batch_size  # priority segment

        # Increase the beta each time we sample a new minibatch
        self.beta = np.amin([1., self.beta + self.beta_increase])  # max = 1

        # Calculate the max_weight
        # p_min = np.amin(self.memory_tree.tree[-self.memory_tree.capacity:]) / self.memory_tree.total_priority
        # if p_min == 0:
        #     p_min = 0.0001
        # max_weight = (p_min * self.batch_size) ** (-self.beta)

        for i in range(self.batch_size):
            """
            A value is uniformly sampled from each range
            """
            a, b = priority_segment * i, priority_segment * (i + 1)

            # This while is to counter that we find a leaf that is not populated yet.
            # It happens when the buffer size is very large.
            data, index, priority = 0, 0, 0
            while data == 0:
                value = np.random.uniform(a, b)

                """
                Experience that corresponds to each value is retrieved
                """
                index, priority, data = self.memory_tree.get_leaf(value)

            # P(i) = p_i**a / sum_k p_k**a
            sampling_probabilities = priority / self.memory_tree.total_priority
            if priority > self.absolute_error_upper:
                debug = 0
            # (1/N * 1/P(i))**b and to normalize it we divide with max_weight
            # So is_weights[i] = (1/N * 1/P(i))**b
            is_weights[i] = np.power(self.batch_size * sampling_probabilities, -self.beta)
            if is_weights[i] > self.absolute_error_upper:
                debug = 0
            idxs[i] = index
            minibatch.append(data)
        is_weights /= is_weights.max()
        return idxs, minibatch, is_weights

    def batch_update(self, idxs, errors):
        """
        Update the priorities on the tree
        """
        errors = errors + self.epsilon
        clipped_errors = np.minimum(errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.alpha)

        for idx, p in zip(idxs, ps):
            self.memory_tree.update(idx, p)


# =========================================  DATA STRUCTURE ======================================================= #

# A binary tree data structure where the parent’s value is the sum of its children
class SumTree:
    """
    This SumTree code is modified version of the code from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Dueling%20Double%20DQN%20with%20PER%20and%20fixed-q%20targets/Dueling%20Deep%20Q%20Learning%20with%20Doom%20(%2B%20double%20DQNs%20and%20Prioritized%20Experience%20Replay).ipynb
    For explanations please see:
    https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/
    """
    data_pointer = 0
    data_length = 0

    def __init__(self, capacity):
        """ Sumtree is a datastructure for sampling faster
        :param capacity: The maximum leafs in the binary tree
        """
        # Initialize the tree with all nodes = 0, and initialize the data with all values = 0

        # Number of leaf nodes (final nodes) that contains experiences
        # Should be power of 2.
        self.capacity = int(capacity)
        assert self.is_power_of_2(self.capacity), "Capacity must be power of 2." + str(capacity)

        # Generate the tree with all nodes values = 0
        # To understand this calculation (2 * capacity - 1) look at the schema above
        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self.tree = np.zeros(2 * capacity - 1)

        """ tree:
            0
           / \
          0   0
         / \ / \
        0  0 0  0  [Size: capacity] it's at this line where the priority scores are stored
        """

        # Contains the experiences (so the size of data is capacity)
        self.data = np.zeros(capacity, dtype=object)

    def __len__(self):
        return self.data_length

    def add(self, data, priority):
        # Look at what index we want to put the experience
        tree_index = self.data_pointer + self.capacity - 1
        # Update data frame
        self.data[self.data_pointer] = data
        # Update the leaf
        self.update(tree_index, priority)
        # Add 1 to data_pointer
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
        if self.data_length < self.capacity:
            self.data_length += 1

    def update(self, tree_index, priority):
        # change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # then propagate the change through tree
        while tree_index != 0:
            """
            Here we want to access the line above
            THE NUMBERS IN THIS TREE ARE THE INDEXES NOT THE PRIORITY VALUES

                0
               / \
              1   2
             / \ / \
            3  4 5  [6] 

            If we are in leaf at index 6, we updated the priority score
            We need then to update index 2 node
            So tree_index = (tree_index - 1) // 2
            tree_index = (6-1)//2
            tree_index = 2 (because // round the result)
            """
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v):
        # Get the leaf_index, priority value of that leaf and experience associated with that index
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for experiences
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_index = 0

        while True:  # the while loop is faster than the method in the reference code
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @staticmethod
    def is_power_of_2(n):
        return ((n & (n - 1)) == 0) and n != 0

    @property
    def total_priority(self):
        return self.tree[0]  # the root

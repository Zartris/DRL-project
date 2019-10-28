import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size

        self.model = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.action_size)
        )

    def forward(self, state):
        """Build a network that maps state -> action values."""
        return self.model(state)


class DuelingQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        self.fc1 = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU()
        )

        # We create two separate streams (both ends with no activation function)
        # One for computing the value function V(s)
        self.value_fc1 = nn.Linear(512, 512)
        self.value_fc2 = nn.Linear(512, 1)

        # The one that calculate A(s,a)
        self.advantage_fc1 = nn.Linear(512, 512)
        self.advantage_fc2 = nn.Linear(512, self.action_size)
        

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.fc1(state)
        # Compute the value function
        value = F.relu(self.value_fc1(x))
        value = self.value_fc2(value)

        # compute the advantage
        advantage = F.relu(self.advantage_fc1(x))
        advantage = self.advantage_fc2(advantage)

        # Compute Q-value:
        
        q = value.expand_as(advantage) + (advantage - advantage.mean(1, keepdim=True).expand_as(advantage))
        return q

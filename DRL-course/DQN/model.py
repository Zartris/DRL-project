import torch
import torch.nn as nn


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
        self.model.apply(self.init_weights)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        return self.model(state)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


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
        self.feature_layer = nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.action_size)
        )
        # Init weights:
        self.feature_layer.apply(self.init_weights)
        self.value_stream.apply(self.init_weights)
        self.advantage_stream.apply(self.init_weights)


    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.feature_layer(state)
        # Compute the value function
        value = self.value_stream(x)

        # compute the advantage
        advantage = self.advantage_stream(x)
        # Compute Q-value:
        q_values = value.expand_as(advantage) + (advantage - advantage.mean(1, keepdim=True).expand_as(advantage))
        return q_values

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

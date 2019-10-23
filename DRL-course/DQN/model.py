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

        "*** YOUR CODE HERE ***"

        self.model = nn.Sequential(
            nn.Linear(state_size, 32),
            nn.Dropout2d(p=0.2, inplace=False),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.Dropout2d(p=0.2, inplace=False),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.Dropout2d(p=0.2, inplace=False),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Dropout2d(p=0.2, inplace=False),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )

    def forward(self, state):
        """Build a network that maps state -> action values."""
        return self.model(state)

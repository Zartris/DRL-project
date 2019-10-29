import torch
from torch import nn


# # Dueling Deep Q-Network (DDQN)
class DDQN(nn.Module):
    def __init__(self, action_size, state_size, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.action_size = action_size
        self.state_size = state_size

        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU()
        )

        # state Value function V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1))

        # Advantage
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, action_size))

        self.feature_layer.apply(self.init_weights)
        self.value_stream.apply(self.init_weights)
        self.advantage_stream.apply(self.init_weights)

    def forward(self, state):
        x = self.feature_layer(state)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q_values = value.expand_as(advantage) + (advantage - advantage.mean(1, keepdim=True).expand_as(advantage))
        return q_values

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

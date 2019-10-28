from torch import nn

# # Dueling Deep Q-Network (DDQN)
class DDQN(nn.Module):
    def __init__(self, action_size, state_size):
        super().__init__()
        self.action_size = action_size
        self.state_size = state_size

        self.fc1 = nn.Sequential(
            nn.Linear(state_size, 1024),
            nn.ReLU()
        )

        # state Value function V(s)
        self.value_layers = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU,
            nn.Linear(1024, 1024),
            nn.ReLU)
        self.value_output = nn.Linear(1024, 1)

        # Advantage
        self.advantage_layers = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU,
            nn.Linear(1024, 1024),
            nn.ReLU)
        self.advantage_output = nn.Linear(1024, action_size)

    def forward(self, state):
        x = self.fc1(state)
        value = self.value_layers(x)
        value = self.value_output(value)

        advantage = self.advantage_layers(x)
        advantage = self.advantage_output(advantage)

        q_values = value.expand_as(advantage) + (advantage - advantage.mean(1, keepdim=True).expand_as(advantage))
        return q_values

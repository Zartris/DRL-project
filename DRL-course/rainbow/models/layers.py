import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# Thanks to https://github.com/cyoon1729/deep-Q-networks/
class FactorizedNoisyLinear(nn.Module):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, in_features, out_features, seed, is_training=True, std_init=0.5, name="noisyLinear"):
        super(FactorizedNoisyLinear, self).__init__()
        self.seed = seed
        torch.manual_seed(self.seed)

        self.in_features = in_features
        self.out_features = out_features
        self.is_training = is_training

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.Tensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.std_init = std_init
        self.name = name

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        """

        :param x:
        :return:
        """
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        y = F.linear(x, weight, bias)

        return y

    def reset_parameters(self):
        std = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-std, std)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))

        self.bias_mu.data.uniform_(-std, std)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

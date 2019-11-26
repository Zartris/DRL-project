import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# Thanks to https://github.com/cyoon1729/deep-Q-networks/
class FactorizedNoisyLinear(nn.Module):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self,
                 in_features,  # Number of input features
                 out_features,  # Number of output features
                 std_init,  # Amount of noise in layer
                 seed=None,  # The env seed (if needed)
                 name="noisyLinear"  # Name for debugging
                 ):
        super(FactorizedNoisyLinear, self).__init__()
        if seed is not None:
            self.seed = seed
            torch.manual_seed(seed)

        self.in_features = in_features
        self.out_features = out_features

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

    def forward(self, x: torch.Tensor):
        """
        Applying noise to the weights and bias to simulate exploring
        :param x: The input state
        :return: Linear translation with noisy weights and bias.
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
        """
        Quote from paper:
        # For factorised noisy networks, each element µi,j was initialised by a sample from an independent
        # uniform distributions U[− (1/ sqrt(p)), + (1/sqrt(p))]
        # and each element σi,j was initialised to a constant sqrt(σ_0/p).
        # The hyperparameter σ_0 is set to 0.5 (Not in our case).
        :return:
        """
        std = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-std, std)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))

        self.bias_mu.data.uniform_(-std, std)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))

    def reset_noise(self):
        """
        Factorised Gaussian noise
        From the paper:
        # Each ε^{w}_{i,j} and ε^{b}_{j} can then be written as:
        ε^{w}_{i,j} = f(ε_i) dot f(ε_j)
        ε^{b}_{j} = f(ε_j)
        where f is a real value function.
        """
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        # ε^{w}_{i,j} = f(ε_i) dot f(ε_j)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in)) # Ger is outer product
        # ε^{b}_{j} = f(ε_j)
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        """
        This is the chosen real value function f:
        sig(x) * sqrt(|x|) as in the paper.
        """
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

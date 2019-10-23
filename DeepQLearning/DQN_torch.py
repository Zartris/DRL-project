from pathlib import Path
from collections import OrderedDict

import torch
from torch import nn
from torchvision import transforms, datasets
import os
import numpy as np


class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_channels=1, input_dims=(210, 160, 4), checkpoint_dir='tmp/dqn'):
        """
        Initializing the deep Q network, taking in images of size (84x84x1).
        :param lr: Is the initial learning rate the network is starting out with.
        :param n_actions: This is the number of action available.
        :param name: This is the name of the network (describing functionality)
        :param input_channels: grayscale = 1, color = 3
        :param input_dims: Is the dimensions of the images we input, (width, height, number_of_img_stacked)
        :param checkpoint_dir: Is the dir we are saving the network to while training.
        """
        super(DeepQNetwork, self).__init__()
        self.lr = lr
        self.n_actions = n_actions
        self.name = name
        self.input_channels = input_channels
        self.input_dims = input_dims
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_file = Path(checkpoint_dir, str(name)+'_deepqnet.ckpt')
        self.current_epoch = 0
        self.build_network()

    def build_network(self):
        """
        :return: the model
        """
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=8, stride=4, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(32 * 28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, self.n_actions)
        )

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        print(x.size())
        #FLAT
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x

    def save_model(self, current_epoch):
        checkpoint = {
            'name': self.name,
            'input_size': self.input_dims,
            'n_actions': self.n_actions,
            'state_dict': self.state_dict(),
            'lr_init': self.lr,
            'current_epoch': current_epoch}

        torch.save(checkpoint, self.checkpoint_file)

    def load_model(self, checkpoint_file=None):
        path = Path(self.checkpoint_file) if checkpoint_file is None else Path(checkpoint_file)
        if not path.exists():
            print(
                "Invalid argument, your model checkpoint file", str(path), "does not existing,",
                "run again and specify specific file in method.")
            return None

        checkpoint = torch.load(str(path))
        self.name = checkpoint["name"]
        self.input_dims = checkpoint["input_dims"]
        self.n_actions = checkpoint["n_actions"]
        state_dict = checkpoint["state_dict"]
        self.load_state_dict(state_dict)
        self.lr = checkpoint["lr_init"]
        self.current_epoch = checkpoint["current_epoch"]

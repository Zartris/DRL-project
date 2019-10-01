from pathlib import Path
import os
import tensorflow as tf
import numpy as np


class DeepQNetwork(object):
    def __init__(self, lr, n_actions, name, fc1_dims=1024, input_dims=(210, 160, 4), checkpoint_dir='tmp/dqn'):
        """
        Initializing the deep Q network.
        :param lr: Is the initial learning rate the network is starting out with.
        :param n_actions: This is TODO::JFS Need this information
        :param name: This is the name of the network (describing functionality)
        :param fc1_dims: (FC = Fully connected)  TODO::JFS Need this information
        :param input_dims: Is the input dimensions of the image we are taking im, where the channel (4 as default) is TODO::JFS Need this information
        :param checkpoint_dir: Is the dir we are saving the network to while training.
        """
        self.lr = lr
        self.n_actions = n_actions
        self.name = name
        self.fc1_dims = fc1_dims
        self.input_dims = input_dims
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(checkpoint_dir, 'deepqnet.ckpt')

        # Tensorflow parameters:
        self.sess = tf.Session()  # Each network need it own session.
        self.saver = tf.train.Saver()  # The saver takes the current sessions and save it to a file.
        self.buil_network()  # Initializing the network (The layout and design)
        # Initializing after the build.
        self.sess.run(tf.global_variables_initializer())
        # Make a hook into all the trainable variables in the network
        self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    def buil_network(self):
        """
        TODO::JFS write the functionality of this class
        :return:
        """
        pass

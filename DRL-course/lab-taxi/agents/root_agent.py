import random

import numpy as np
from collections import defaultdict

from agents.base_agent import BaseAgent
from agents.deliver_agent import DeliverAgent
from agents.move_agent import MoveAgent
from agents.pickup_agent import PickupAgent


class RootAgent(BaseAgent):
    def __init__(self, name="Root", nA=2, alpha=0.5, gamma=1, epsilon_init=0.5, epsilon_decay=0.999,
                 epsilon_limit=0.001, sarsa="EXPECTED"):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        super().__init__(name, nA, alpha, gamma, epsilon_init, epsilon_decay, epsilon_limit, sarsa)
        mover = MoveAgent()
        self.pickup = PickupAgent(mover)
        self.deliver = DeliverAgent(mover)

    def play_episode(self, env):
        """
        apparently this is the bottle neck in this implementation.
        The counter is used to bypass the bottle neck, but should not be there.
        :param env:
        :return: reward, env
        """
        state = env.reset()
        samp_reward = 0
        counter = 0
        while True:
            # agent selects an action
            action = self.select_action(state)
            action = counter
            # agent performs the selected action
            next_state, reward, done, env_new = self.perform_action(action, env, state)
            env = env_new
            # agent performs internal updates based on sampled experience
            self.step(state, action, reward, next_state, done)
            # update the sampled reward
            samp_reward += reward
            # update the state (s <- s') to next time step
            state = next_state
            counter += 1
            if done:
                # save final sampled reward
                break
        return samp_reward, env

    def perform_action(self, action, env, state):
        """

        :param action:
        :param env:
        :param state:
        :return:  next_state, reward, done, env
        """
        if action == 0:  # PICKUP
            return self.pickup.perform_task(env, state)
        else:  # Deliver
            return self.deliver.perform_task(env, state)

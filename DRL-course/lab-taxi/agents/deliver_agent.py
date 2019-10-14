import random

import numpy as np
from collections import defaultdict

from agents.base_agent import BaseAgent
from agents.move_agent import MoveAgent


class DeliverAgent(BaseAgent):
    def __init__(self, mover_agent, name="deliver", nA=2, alpha=1, gamma=1, epsilon_init=0.5, epsilon_decay=0.99995,
                 epsilon_limit=0.0001, sarsa="MAX"):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        super().__init__(nA, alpha, gamma, epsilon_init, epsilon_decay, epsilon_limit, sarsa)
        self.mover_agent = mover_agent

    def perform_task(self, env, state):
        samp_reward = 0
        while True:
            # agent selects an action
            action = self.select_action(state)
            # agent performs the selected action
            next_state, reward, done, env_new = self.perform_action(action, env, state)
            env = env_new
            # agent performs internal updates based on sampled experience
            self.step(state, action, reward, next_state, done)
            # update the sampled reward
            samp_reward += reward
            # update the state (s <- s') to next time step
            state = next_state
            if done:
                # save final sampled reward
                break
        return samp_reward, env, done

    def perform_action(self, action, env, state):
        """
        Perform action.
        :param action:
        :param env:
        :param state:
        :return: next_state, reward, env
        """
        if action == 0:  # Deliver
            next_state, reward, done, _ = env.step(4)
            return next_state, reward, env
        else:  # Move
            taxi_row, taxi_col, pass_idx, dest_idx = self.decode_state(state)
            position = self.color_to_pos(pass_idx)
            return self.mover_agent.move_to(position, state, env)

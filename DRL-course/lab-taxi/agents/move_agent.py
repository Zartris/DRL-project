import random

import numpy as np
from collections import defaultdict
from agents.base_agent import BaseAgent


class MoveAgent(BaseAgent):
    def __init__(self, name="Mover", nA=4, alpha=1, gamma=1, epsilon_init=0.5, epsilon_decay=0.99995,
                 epsilon_limit=0.0001, sarsa="MAX"):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        super().__init__(nA, alpha, gamma, epsilon_init, epsilon_decay, epsilon_limit, sarsa)

    def move_to(self, position, state, env):
        goal_row, goal_col = position
        samp_reward = 0
        while True:
            # agent selects an action
            action = self.select_action(state)
            # agent performs the selected action
            next_state, reward, done, _ = env.step(action)

            taxi_row, taxi_col, _, _ = self.decode_state(state)
            is_moving_done = taxi_row == goal_row and taxi_col == goal_col
            # agent performs internal updates based on sampled experience
            self.step(state, action, reward, next_state, is_moving_done)
            # update the sampled reward
            samp_reward += reward
            # update the state (s <- s') to next time step
            state = next_state

            if is_moving_done:
                # save final sampled reward
                break
        return samp_reward, state, env

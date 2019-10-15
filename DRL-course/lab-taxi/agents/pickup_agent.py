from agents.base_agent import BaseAgent


class PickupAgent(BaseAgent):
    def __init__(self, mover_agent, name="pickup", nA=2, alpha=1, gamma=1, epsilon_init=0.5, epsilon_decay=0.999,
                 epsilon_limit=0.0001, sarsa="EXPECTED"):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        super().__init__(name, nA, alpha, gamma, epsilon_init, epsilon_decay, epsilon_limit, sarsa)
        self.mover_agent = mover_agent

    def perform_task(self, env, state):
        """

        :param env:
        :param state:
        :return: state, samp_reward, done, env
        """
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
        return state, samp_reward, False, env

    def perform_action(self, action, env, state):
        """
        Perform action.
        :param action:
        :param env:
        :param state:
        :return: next_state, reward, done, env
        """
        if action == 0:  # PICKUP
            next_state, reward, done, _ = env.step(4)
            taxi_row, taxi_col, pass_idx, dest_idx = self.decode_state(next_state)
            return next_state, reward, pass_idx == 4, env
        else:  # Move
            taxi_row, taxi_col, pass_idx, dest_idx = self.decode_state(state)
            position = self.color_to_pos(pass_idx)
            if position == (-1, -1):
                next_state, reward, done, _ = env.step(4)
                taxi_row, taxi_col, pass_idx, dest_idx = self.decode_state(next_state)
                return next_state, reward, pass_idx == 4, env
            return self.mover_agent.move_to(position, state, env)

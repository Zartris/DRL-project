import sys

import numpy as np

import Q_table_tiling


class QLearningAgent:
    def __init__(self, env, nA, tiling_specs, epsilon, alpha, gamma,
                 epsilon_decay=0.9995, epsilon_limit=0.01,
                 alpha_decay=0.999, alpha_limit=0.01):
        self.env = env
        self.nA = nA
        self.tiling_specs = tiling_specs
        self.tiled_QTable = Q_table_tiling.TiledQTable(env.observation_space.low, env.observation_space.high,
                                                       tiling_specs, nA)
        self.eps = epsilon
        self.eps_init = epsilon
        self.eps_limit = epsilon_limit
        self.eps_decay = epsilon_decay

        self.alpha = alpha
        self.alpha_limit = alpha_limit
        self.alpha_decay = alpha_decay

        self.gamma = gamma

    def train(self, env, num_episodes=20000, mode='train'):
        """Run agent in given reinforcement learning environment and return scores."""
        scores = []
        max_avg_score = -np.inf
        for i_episode in range(1, num_episodes + 1):
            # Initialize episode
            state = env.reset()
            action = self.reset_episode(state)
            total_reward = 0
            done = False

            # Roll out steps until done
            while not done:
                state, reward, done, info = env.step(action)
                total_reward += reward
                action = self.act(state, reward, done, mode)

            # Save final score
            scores.append(total_reward)

            # Print episode stats
            if mode == 'train':
                if len(scores) > 100:
                    avg_score = np.mean(scores[-100:])
                    if avg_score > max_avg_score:
                        max_avg_score = avg_score

                if i_episode % 100 == 0:
                    print("\rEpisode {}/{} | Max Average Score: {}".format(i_episode, num_episodes, max_avg_score),
                          end="")
                    sys.stdout.flush()

        return scores

    def reset_episode(self, state):
        """Reset variables for a new episode."""
        # Gradually decrease exploration rate
        self.eps *= self.eps_decay
        self.eps = max(self.eps, self.eps_limit)

        # Decide initial action
        self.last_state = state
        self.last_action = np.argmax([self.tiled_QTable.get(state, a) for a in range(self.nA)])
        return self.last_action

    def reset_exploration(self, epsilon=None):
        """Reset exploration rate used when training."""
        self.eps = epsilon if epsilon is not None else self.eps_init

    def act(self, state, reward=None, done=None, mode='train'):
        """Pick next action and update internal Q table (when mode != 'test')."""
        q_values = [self.tiled_QTable.get(state, a) for a in range(self.nA)]
        if mode == 'test':
            # Test mode: Simply produce an action
            action = np.argmax(q_values)
        else:
            # Train mode (default): Update Q table, pick next action
            # Note: We update the Q table entry for the *last* (state, action) pair with current state, reward
            value = reward + self.gamma * max(q_values)
            self.tiled_QTable.update(self.last_state, self.last_action, value, self.alpha)
            # Exploration vs. exploitation
            do_exploration = np.random.uniform(0, 1) < self.eps
            if do_exploration:
                # Pick a random action
                action = np.random.randint(0, self.nA)
            else:
                # Pick the best action from Q table
                action = np.argmax(q_values)

        # Roll over current state, action for next step
        self.last_state = state
        self.last_action = action
        return action

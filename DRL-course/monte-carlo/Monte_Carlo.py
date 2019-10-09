## OVERUSE OF COMMENTS - but it is for learning so it's okay

import sys
from collections import defaultdict

import gym
import numpy as np
from plot_utils import plot_blackjack_values, plot_policy

env = gym.make('Blackjack-v0')


def generate_episode_from_Q(bj_env, Q, epsilon, nA, i_episode, DEBUG=False):
    episode = []
    state = bj_env.reset()
    if DEBUG:
        print("\nStarting episode", str(i_episode))
    while True:

        # If policy contains the state, use that action probability else chose random.
        if state in Q:
            action = np.random.choice(np.arange(nA), p=greedyQ_policy_to_state(Q[state], epsilon, nA))
        else:
            action = bj_env.action_space.sample()

        next_state, reward, done, info = bj_env.step(action)
        if DEBUG:
            action_str, reward_str = action_and_reward_to_str(action, reward, done)
            print(state, action_str, reward_str)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode


def action_and_reward_to_str(action, reward, done):
    if action == 0:
        action_str = "STICK"
    else:
        action_str = "HIT"
    if reward == 0 and done:
        reward_str = "Draw"
    elif reward == 0:
        reward_str = "Keep going"
    elif reward == 1:
        reward_str = "WIN"
    else:
        reward_str = "LOST"
    return action_str, reward_str


# For notes:
def npmax(l):
    max_idx = np.argmax(l)
    max_val = l[max_idx]
    return max_idx, max_val


def greedyQ_policy_to_state(Q_state, epsilon, nA):
    """
    Finding the new policy corresponding to the Q-table (at current state)
    :param Q_state: The current row in the Q table corresponding to the state we are in now.
    :param epsilon: The current value of exploration vs exploitation
    :param nA: Number of actions
    :return:  list of probabilities of choosing an action.
    """

    # Setting all action to have the same change of being chosen.
    policy_state = np.ones(nA) * (epsilon / nA)

    # Finding the index with the highest value (This is the best action to take.)
    best_action = np.argmax(Q_state)

    # Reserving the rest of the probability to the best (current best) action.
    policy_state[best_action] = (1 - epsilon) + (epsilon / nA)

    return policy_state


def mc_control(env, num_episodes, alpha, gamma=1, epsilon_init=1, epsilon_decay=.9999, epsilon_limit=0.1):
    """
    Monte carlo (MC) control is the main loop for learning and filling out the Q-table.
    :param env: The environment we are trying to learn from.
    :param num_episodes: How many episodes we are willing to go through
    :param alpha: Learning rate
    :param gamma: Is the discount rate we use for prioritizing the reward not over the rewards in the future.
    :param epsilon_init: Start exploring rate, (The higher, the more we explore)
    :param epsilon_decay: The decay from exploring to exploiting
    :param epsilon_limit: Don't ever stop exploring.
    :return: The optimal policy and Q table
    """
    nA = env.action_space.n
    # initialize empty dictionary of arrays
    Q = defaultdict(lambda: np.zeros(nA))
    epsilon = epsilon_init
    # loop over episodes
    for i_episode in range(1, num_episodes + 1):
        # monitor progress
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        seen_states = []
        # Generate an episode with current policy.
        episode = generate_episode_from_Q(env, Q, epsilon, nA, i_episode)

        states, actions, rewards = zip(*episode)
        # Prepare for discount:
        discount = np.array([gamma ** i for i in range(len(states) + 1)])

        # Time to learn from this episode:
        for index, state in enumerate(states):
            if state not in seen_states:
                seen_states.append(state)
                # Noted accumulated rewards
                G_t = sum(rewards[index:] * discount[:-(index + 1)])
                # Delta is the error between what we predicted Q(s,a) and the accumulated reward
                old_Q = Q[state][actions[index]]
                delta = G_t - old_Q
                Q[state][actions[index]] = old_Q + alpha * delta

        # Update epsilon for next iteration
        # if epsilon <= epsilon_limit:
        #     epsilon = epsilon_limit
        # else:
        #     epsilon = epsilon * epsilon_decay
        epsilon = max(epsilon * epsilon_decay, epsilon_limit)
    policy = dict((state, np.argmax(action_list)) for (state, action_list) in Q.items())
    return policy, Q


if __name__ == '__main__':
    # obtain the estimated optimal policy and action-value function
    policy, Q = mc_control(env, num_episodes=5000000, alpha=0.03)
    # obtain the corresponding state-value function
    V = dict((k, np.max(v)) for k, v in Q.items())

    # plot the state-value function
    plot_blackjack_values(V)
    # plot the policy
    plot_policy(policy)

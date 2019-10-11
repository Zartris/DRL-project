import random
import sys
import gym
import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt

import check_test
from plot_utils import plot_values


def update_Q_sarsa(Q, state, action, reward, alpha, gamma, next_state, next_action):
    current = Q[state][action]
    Qsa_next = Q[next_state][next_action] if next_state is not None else 0
    target = reward + (gamma * Qsa_next)
    return current + (alpha * (target - current))


def update_Q_sarsa_expected(Q, state, action, reward, alpha, gamma, next_state, epsilon, nA):
    current = Q[state][action]  # estimate in Q-table (for current state, action pair)
    policy_s = np.ones(nA) * epsilon / nA  # current policy (for next state S')
    policy_s[np.argmax(Q[next_state])] = 1 - epsilon + (epsilon / nA)  # greedy action
    Qsa_next = np.dot(Q[next_state], policy_s)  # get value of state at next time step
    target = reward + (gamma * Qsa_next)  # construct target
    return current + (alpha * (target - current))  # get updated value


def get_action_epsilon_greedy(Q, state, nA, epsilon):
    # If policy contains the state, use that action probability else chose random.
    if state in Q:
        action = np.random.choice(np.arange(nA), p=greedyQ_policy_to_state(Q[state], epsilon, nA))
    else:
        action = env.action_space.sample()
    return action


def greedyQ_policy_to_state(Q_state, epsilon, nA):
    # Creating a slot for each action
    policy_s = np.ones(nA)
    # Fill all slots with default action probability
    policy_s.fill(epsilon / nA)
    # The best action is given higher probability
    policy_s[np.argmax(Q_state)] = 1 - epsilon + (epsilon / nA)
    return policy_s


def epsilon_greedy(Q, state, nA, eps):
    """Selects epsilon-greedy action for supplied state.

    Params
    ======
        Q (dictionary): action-value function
        state (int): current state
        nA (int): number actions in the environment
        eps (float): epsilon
    """
    if random.random() > eps:  # select greedy action with probability epsilon
        return np.argmax(Q[state])
    else:  # otherwise, select an action randomly
        return random.choice(np.arange(env.action_space.n))


def sarsa(env, num_episodes, alpha, gamma=1.0, epsilon_init=1, epsilon_decay=.999, epsilon_limit=.02):
    # initialize action-value function (empty dictionary of arrays)
    nA = env.action_space.n  # number of actions
    Q = defaultdict(lambda: np.zeros(nA))
    # initialize performance monitor
    # epsilon = epsilon_init
    # loop over episodes
    for i_episode in range(1, num_episodes + 1):
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        epsilon = 1.0 / i_episode
        Q = generate_episode_sarsa(i_episode, Q, env, env.nA, epsilon, alpha, gamma)

        # Update epsilon:
        # epsilon = max(epsilon * epsilon_decay, epsilon_limit)

    return Q


def generate_episode_sarsa(i_episode, Q, env, nA, epsilon, alpha, gamma, DEBUG=False):
    state = env.reset()
    action = epsilon_greedy(Q, state, nA, epsilon)
    if DEBUG:
        print("\nStarting episode", str(i_episode))
    while True:
        # Find next state given the action
        next_state, reward, done, info = env.step(action)
        # Chose next action
        if not done:
            next_action = epsilon_greedy(Q, next_state, nA, epsilon)
            # Update Q-table
            Q[state][action] = update_Q_sarsa(Q, state, action, reward, alpha, gamma, next_state, next_action)
            # episode.append((state, action, reward))
            state = next_state
            action = next_action
        if done:
            Q[state][action] = update_Q_sarsa(Q, state, action, reward, alpha, gamma, None, None)
            break
    return Q


def q_learning(env, num_episodes, alpha, gamma=1.0, plot_every=100):
    # initialize empty dictionary of arrays
    nA = env.action_space.n
    Q = defaultdict(lambda: np.zeros(nA))

    tmp_scores = deque(maxlen=plot_every)  # deque for keeping track of scores
    avg_scores = deque(maxlen=num_episodes)  # average scores over every plot_every episodes

    # loop over episodes
    for i_episode in range(1, num_episodes + 1):
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        score = 0
        state = env.reset()
        epsilon = 1.0 / i_episode
        action = epsilon_greedy(Q, state, nA, epsilon)
        while True:
            # Find next state given the action
            next_state, reward, done, info = env.step(action)
            score += reward
            # Chose next action
            if not done:
                next_action = epsilon_greedy(Q, next_state, nA, epsilon)
                max_action = np.argmax(Q[next_state])
                # Update Q-table
                Q[state][action] = update_Q_sarsa(Q, state, action, reward, alpha, gamma, next_state, max_action)
                # episode.append((state, action, reward))
                state = next_state
                action = next_action
            if done:
                Q[state][action] = update_Q_sarsa(Q, state, action, reward, alpha, gamma, None, None)
                tmp_scores.append(score)
                break
        if (i_episode % plot_every == 0):
            avg_scores.append(np.mean(tmp_scores))

        plot(num_episodes, avg_scores, plot_every)
    return Q


def expected_sarsa(env, num_episodes, alpha, gamma=1.0, plot_every=100):
    # initialize empty dictionary of arrays
    nA = env.action_space.n
    Q = defaultdict(lambda: np.zeros(env.nA))

    # plot
    tmp_scores = deque(maxlen=plot_every)  # deque for keeping track of scores
    avg_scores = deque(maxlen=num_episodes)  # average scores over every plot_every episodes

    # loop over episodes
    for i_episode in range(1, num_episodes + 1):
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        score = 0

        # Have to be static since we are using this expected value.
        epsilon = 0.005

        state = env.reset()

        while True:
            action = epsilon_greedy(Q, state, nA, epsilon)
            next_state, reward, done, info = env.step(action)
            score += reward
            if not done:
                Q[state][action] = update_Q_sarsa_expected(Q, state, action, reward, alpha, gamma, next_state, epsilon,
                                                           nA)
                state = next_state
            else:
                tmp_scores.append(score)
                break

        if i_episode % plot_every == 0:
            avg_scores.append(np.mean(tmp_scores))

    plot(num_episodes, avg_scores, plot_every)
    return Q


def plot(num_episodes, avg_scores, plot_every):
    plt.plot(np.linspace(0, num_episodes, len(avg_scores), endpoint=False), np.asarray(avg_scores))
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)
    plt.show()
    # print best 100-episode performance
    print(('Best Average Reward over %d Episodes: ' % plot_every), np.max(avg_scores))


if __name__ == '__main__':
    env = gym.make('CliffWalking-v0')
    print(env.action_space)
    print(env.observation_space)
    V_opt = np.zeros((4, 12))
    V_opt[0:13][0] = -np.arange(3, 15)[::-1]
    V_opt[0:13][1] = -np.arange(3, 15)[::-1] + 1
    V_opt[0:13][2] = -np.arange(3, 15)[::-1] + 2
    V_opt[3][0] = -13
    # plot_values(V_opt)
    sarsa_m = False
    sarsa_max = False
    sarsa_expected = True
    if sarsa_m:
        # obtain the estimated optimal policy and corresponding action-value function
        Q_sarsa = sarsa(env, 50000, .01)

        # print the estimated optimal policy
        policy_sarsa = np.array([np.argmax(Q_sarsa[key]) if key in Q_sarsa else -1 for key in np.arange(48)]).reshape(4,
                                                                                                                      12)
        check_test.run_check('td_control_check', policy_sarsa)
        print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
        print(policy_sarsa)

        # plot the estimated optimal state-value function
        V_sarsa = ([np.max(Q_sarsa[key]) if key in Q_sarsa else 0 for key in np.arange(48)])
        plot_values(V_sarsa)
    if sarsa_max:
        Q_sarsamax = q_learning(env, 5000, .01)

        # print the estimated optimal policy
        policy_sarsamax = np.array(
            [np.argmax(Q_sarsamax[key]) if key in Q_sarsamax else -1 for key in np.arange(48)]).reshape((4, 12))
        check_test.run_check('td_control_check', policy_sarsamax)
        print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
        print(policy_sarsamax)

        # plot the estimated optimal state-value function
        plot_values([np.max(Q_sarsamax[key]) if key in Q_sarsamax else 0 for key in np.arange(48)])
    if sarsa_expected:
        # obtain the estimated optimal policy and corresponding action-value function
        Q_expsarsa = expected_sarsa(env, 10000, 1)

        # print the estimated optimal policy
        policy_expsarsa = np.array(
            [np.argmax(Q_expsarsa[key]) if key in Q_expsarsa else -1 for key in np.arange(48)]).reshape(4, 12)
        check_test.run_check('td_control_check', policy_expsarsa)
        print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
        print(policy_expsarsa)

        # plot the estimated optimal state-value function
        plot_values([np.max(Q_expsarsa[key]) if key in Q_expsarsa else 0 for key in np.arange(48)])

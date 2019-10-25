#!/usr/bin/env python
# coding: utf-8

# # Deep Q-Network (DQN)
# ---
# In this notebook, you will implement a DQN agent with OpenAI Gym's LunarLander-v2 environment.
# 
# ### 1. Import the Necessary Packages

# In[ ]:
import time

import gym

from agents.double_dqn_agent import DoubleDQNAgent, DoubleDQNAgentPER
from agents.dqn_agent import DQNAgent, DQNAgentPER
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

is_ipython = 'inline' in plt.get_backend()
if is_ipython:
    from IPython import display

plt.ion()


def dqn(agent, scheduler=None, save_file='checkpoint.pth', n_episodes=2000000, max_t=1000, eps_start=1.0, eps_end=0.01,
        eps_decay=0.995):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    time_window = deque(maxlen=10)  # last 100 scores
    eps = eps_start  # initialize epsilon
    best_avg = 280.0
    for i_episode in range(1, n_episodes + 1):

        state = env.reset()
        score = 0
        start = time.time()
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        time_window.append(time.time() - start)
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        if scheduler is not None:
            scheduler.step(np.mean(scores_window), i_episode)
        print('\rEpisode {}\tAverage Score: {:.2f}\tAverage Time pr episode {:.2f} seconds'.format(i_episode,
                                                                                              np.mean(scores_window),
                                                                                              np.mean(time_window)),
              end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tTime left {:.2f} seconds'.format(i_episode, np.mean(scores_window),
                                                                                    np.mean(time_window) * (
                                                                                            n_episodes - i_episode)))
        if np.mean(scores_window) >= best_avg:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}\tTime left {:.2f} seconds'.format(
                i_episode - 100,
                np.mean(scores_window), np.mean(time_window) * (n_episodes - i_episode)))
            torch.save(agent.qnetwork_local.state_dict(), str(save_file))
            best_avg = np.mean(scores_window)
    return scores


if __name__ == '__main__':
    stop = False
    env = gym.make('LunarLander-v2')
    env.seed(0)
    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.n)

    test_vanilla_DQN = False
    test_double_DQN = False
    test_DQN_PER = True
    test_double_DQN_PER = False

    if test_vanilla_DQN:
        name = "DQNAgent"
        agent = DQNAgent(state_size=8, action_size=4, seed=0)
        save_file = "DQNAgent_checkpoint.pth"
    elif test_double_DQN:
        name = "DoubleDQNAgent"
        agent = DoubleDQNAgent(state_size=8, action_size=4, seed=0)
        save_file = "DoubleDQNAgent_checkpoint.pth"
    elif test_DQN_PER:
        name = "DQNAgentPER"
        agent = DQNAgentPER(state_size=8, action_size=4, seed=0)
        save_file = "DQNAgentPER_checkpoint.pth"
    elif test_double_DQN_PER:
        name = "DoubleDQNAgentPER"
        agent = DoubleDQNAgentPER(state_size=8, action_size=4, seed=0)
        save_file = "DoubleDQNAgentPER_checkpoint.pth"
    else:
        print("Pick an agent")
        stop = True

    if not stop:
        print("device used", agent.device)
        test_untrained_agent = False
        train_agent = True

        # watch an untrained agent
        if test_untrained_agent:
            state = env.reset()
            # img = plt.imshow(env.render(mode='rgb_array'))
            for j in range(200):
                action = agent.act(state)
                # img.set_data(env.render(mode='rgb_array'))
                # plt.axis('off')
                env.render()
                state, reward, done, _ = env.step(action)
                if done:
                    break

            env.close()

        if train_agent:
            print("train", str(name))
            scores = dqn(agent, save_file=save_file, n_episodes=1000)

            # plot the scores
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.plot(np.arange(len(scores)), scores)
            plt.ylabel('Score')
            plt.xlabel('Episode #')
            plt.show()
            # Load best
            agent.qnetwork_local.load_state_dict(torch.load(save_file))

            for i in range(3):
                state = env.reset()
                # img = plt.imshow(env.render(mode='rgb_array'))
                for j in range(200):
                    action = agent.act(state)
                    # img.set_data(env.render(mode='rgb_array'))
                    # plt.axis('off')
                    # display.display(plt.gcf())
                    # display.clear_output(wait=True)
                    env.render()
                    state, reward, done, _ = env.step(action)
                    if done:
                        break

            env.close()

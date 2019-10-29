#!/usr/bin/env python
# coding: utf-8

# # Deep Q-Network (DQN)
# ---
# In this notebook, you will implement a DQN agent with OpenAI Gym's LunarLander-v2 environment.
# 
# ### 1. Import the Necessary Packages

# In[ ]:
import os
import time
from collections import deque

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from agents.double_dqn_agent import DoubleDQNAgent, DoubleDQNAgentPER
from agents.dqn_agent import DQNAgent, DQNAgentPER
from model import QNetwork, DuelingQNetwork

is_ipython = 'inline' in plt.get_backend()
if is_ipython:
    pass

plt.ion()


def dqn(agent, score_file, scheduler=None, save_file='checkpoint.pth', n_episodes=2000000, max_t=1000, eps_start=1.0,
        eps_end=0.01,
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
    best_avg = 200.0

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
                                                                                                   np.mean(
                                                                                                       scores_window),
                                                                                                   np.mean(
                                                                                                       time_window)),
              end="")

        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tTime left {:.2f} seconds'.format(i_episode,
                                                                                         np.mean(scores_window),
                                                                                         np.mean(time_window) * (
                                                                                                 n_episodes - i_episode)))
            score_file.write('\tEpisode {}\tAverage Score: {:.2f}\n'.format(i_episode, np.mean(scores_window)))
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
    state_size = 8
    action_size = 4
    seed = 0
    test_DQN_model = False
    test_Dueling_DQN_model = True
    models = []
    if test_DQN_model:
        model = (QNetwork(state_size=state_size, action_size=action_size, seed=seed),
                 QNetwork(state_size=state_size, action_size=action_size, seed=seed))
        models.append(("QNetwork", model))
    if test_Dueling_DQN_model:
        model = (DuelingQNetwork(state_size=state_size, action_size=action_size, seed=seed),
                 DuelingQNetwork(state_size=state_size, action_size=action_size, seed=seed))
        models.append(("DuelingQNetwork", model))

    test_vanilla_DQN = False
    test_double_DQN = False
    test_DQN_PER = True
    test_double_DQN_PER = True

    testing_pairs = []
    for model_name, model in models:
        if test_vanilla_DQN:
            name = "Model: " + model_name + ", Agent: DQNAgent"
            agent = DQNAgent(state_size=state_size, action_size=action_size, seed=seed, models=model)
            save_file = model_name + "_DQNAgent_checkpoint.pth"
            testing_pairs.append((name, agent, save_file))
        if test_double_DQN:
            name = "Model: " + model_name + ", Agent: DoubleDQNAgent"
            agent = DoubleDQNAgent(state_size=state_size, action_size=action_size, seed=seed, models=model)
            save_file = model_name + "_DoubleDQNAgent_checkpoint.pth"
            testing_pairs.append((name, agent, save_file))
        if test_DQN_PER:
            name = "Model: " + model_name + ", Agent: DQNAgentPER"
            agent = DQNAgentPER(state_size=state_size, action_size=action_size, seed=seed, models=model)
            save_file = model_name + "_DQNAgentPER_checkpoint.pth"
            testing_pairs.append((name, agent, save_file))
        if test_double_DQN_PER:
            name = "Model: " + model_name + ", Agent: DoubleDQNAgentPER"
            agent = DoubleDQNAgentPER(state_size=state_size, action_size=action_size, seed=seed, models=model)
            save_file = model_name + "_DoubleDQNAgentPER_checkpoint.pth"
            testing_pairs.append((name, agent, save_file))

    print("device used", agent.device)
    test_untrained_agent = False
    train_agent = True
    show_result = False

    # watch an untrained agent
    if test_untrained_agent:
        os.remove("scores_tmp.md")
        f = open("scores_tmp.md", "a+")
        f.write("\n## " + str("name") + "\n")
        for i in range(10):
            f.write(str(i))
        f.close()
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
        # os.remove("scores_tmp.md")

        for name, agent, save_file in testing_pairs:
            # We open and close the filestream inside for-loop
            # to insure we write the result even if we stop before finish
            f = open("scores_tmp.md", "a+")
            print("train", str(name))
            f.write("\n## " + str(name) + "\n")
            scores = dqn(agent=agent, score_file=f, save_file=save_file, n_episodes=1000)
            if show_result:
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
            f.close()

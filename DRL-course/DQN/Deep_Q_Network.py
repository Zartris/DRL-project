#!/usr/bin/env python
# coding: utf-8

# # Deep Q-Network (DQN)
# ---
# In this notebook, you will implement a DQN agent with OpenAI Gym's LunarLander-v2 environment.
# 
# ### 1. Import the Necessary Packages

# In[ ]:


import gym

from double_dqn_agent import DoubleDQNAgent
from dqn_agent import Agent
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

is_ipython = 'inline' in plt.get_backend()
if is_ipython:
    from IPython import display

plt.ion()


def dqn(agent, scheduler=None, n_episodes=2000000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
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
    eps = eps_start  # initialize epsilon
    best_avg = 280.0
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        if scheduler is not None:
            scheduler.step(np.mean(scores_window), i_episode)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= best_avg:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            best_avg = np.mean(scores_window)
    return scores


if __name__ == '__main__':
    ex1 = False
    ex2 = False
    ex3 = False
    ex4 = True
    # ### 2. Instantiate the Environment and Agent
    #
    # Initialize the environment in the code cell below.

    # In[ ]:

    env = gym.make('LunarLander-v2')
    env.seed(0)
    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.n)

    # Before running the next code cell, familiarize yourself with the code in **Step 2** and **Step 3** of this notebook, along with the code in `dqn_agent.py` and `model.py`.  Once you have an understanding of how the different files work together,
    # - Define a neural network architecture in `model.py` that maps states to action values.  This file is mostly empty - it's up to you to define your own deep Q-network!
    # - Finish the `learn` method in the `Agent` class in `dqn_agent.py`.  The sampled batch of experience tuples is already provided for you; you need only use the local and target Q-networks to compute the loss, before taking a step towards minimizing the loss.
    #
    # Once you have completed the code in `dqn_agent.py` and `model.py`, run the code cell below.  (_If you end up needing to make multiple changes and get unexpected behavior, please restart the kernel and run the cells from the beginning of the notebook!_)
    #
    # You can find the solution files, along with saved model weights for a trained agent, in the `solution/` folder.  (_Note that there are many ways to solve this exercise, and the "solution" is just one way of approaching the problem, to yield a trained agent._)

    # In[ ]:

    # watch an untrained agent
    if ex1:
        agent = Agent(state_size=8, action_size=4, seed=0)
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

    # ### 3. Train the Agent with DQN
    #
    # Run the code cell below to train the agent from scratch.  You are welcome to amend the supplied values of the parameters in the function, to try to see if you can get better performance!

    # In[ ]:
    if ex2:
        agent = Agent(state_size=8, action_size=4, seed=0)
        agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
        agent.qnetwork_target.load_state_dict(torch.load('checkpoint.pth'))
        scheduler = agent.scheduler

        scores = dqn(agent, scheduler=scheduler)

        # plot the scores
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()

        # ### 4. Watch a Smart Agent!
        #
        # In the next code cell, you will load the trained weights from file to watch a smart agent!

        # In[ ]:

        # load the weights from file

        agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

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
    if ex3:
        agent = Agent(state_size=8, action_size=4, seed=0)
        agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

        for i in range(10):
            state = env.reset()
            for j in range(500):
                action = agent.act(state)

                env.render()
                state, reward, done, _ = env.step(action)
                if done:
                    break

        env.close()
    if ex4:
        agent = DoubleDQNAgent(state_size=8, action_size=4, seed=0)
        # agent.qnetwork_local.load_state_dict(torch.load('checkpoint_double.pth'))
        # agent.qnetwork_target.load_state_dict(torch.load('checkpoint_double.pth'))
        # scheduler = agent.scheduler

        scores = dqn(agent)

        # plot the scores
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()

        # ### 4. Watch a Smart Agent!
        #
        # In the next code cell, you will load the trained weights from file to watch a smart agent!

        # In[ ]:

        # load the weights from file

        agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

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


    # ### 5. Explore
    #
    # In this exercise, you have implemented a DQN agent and demonstrated how to use it to solve an OpenAI Gym environment.  To continue your learning, you are encouraged to complete any (or all!) of the following tasks:
    # - Amend the various hyperparameters and network architecture to see if you can get your agent to solve the environment faster.  Once you build intuition for the hyperparameters that work well with this environment, try solving a different OpenAI Gym task with discrete actions!
    # - You may like to implement some improvements such as prioritized experience replay, Double DQN, or Dueling DQN!
    # - Write a blog post explaining the intuition behind the DQN algorithm and demonstrating how to use it to solve an RL environment of your choosing.

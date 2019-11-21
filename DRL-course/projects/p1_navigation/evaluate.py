import time
from collections import deque

import numpy as np
import torch

from projects.p1_navigation.utils import helper


def evaluate(agent, brain_name, test_env, n_episodes, train_episode="Loaded model", model_save_file=None,
             current_best=10000, set_fast_mode=True):
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    time_window = deque(maxlen=10)  # last 10 iter
    agent.model.eval()
    for i_episode in range(1, n_episodes + 1):
        state = test_env.reset(train_mode=set_fast_mode)[brain_name].vector_observations[0]
        score = 0
        start = time.time()
        max_reached = False
        while not max_reached:
            action = int(agent.act(state))
            next_state, reward, done, max_reached = helper.unpack_braininfo(brain_name, test_env.step(action))
            state = next_state
            score += reward
            if done:
                break
        time_window.append(time.time() - start)
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        print(
            '\rTest: Episode {}\tAverage Score: {:.2f}\tthis Score: {:.2f}\tAverage Time pr episode {:.2f} seconds'.format(
                i_episode,
                np.mean(
                    scores_window),
                score,
                np.mean(
                    time_window)),
            end="")
        if i_episode % 100 == 0:
            print('Test: \rEpisode {}\tAverage Score: {:.2f}\tTime left {:.2f} seconds'.format(i_episode,
                                                                                               np.mean(scores_window),
                                                                                               np.mean(time_window) * (
                                                                                                       n_episodes - i_episode)))
            if np.mean(scores_window) >= current_best:
                if model_save_file != None:
                    torch.save(agent.model.state_dict(), str(model_save_file))
                    current_best = np.mean(scores_window)
    agent.model.train()
    return '\n\ttrain_episode: {}\t Average Score over {} episodes: {}'.format(str(train_episode), str(n_episodes),
                                                                               np.mean(scores)), current_best

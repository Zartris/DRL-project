import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np

from projects.p1_navigation.evaluate import evaluate
from projects.p1_navigation.utils import helper


def train(agent, brain_name, train_env, file, save_img="plot.png", save_file='checkpoint.pth',
          n_episodes=2000000, evaluation_interval=200, plot=False, plot_title="title"):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    if plot:
        buffer = 1
        min_score = 0
        max_score = min_score + buffer
        fig = plt.figure()
        score_ax = fig.add_subplot(111)
        score_line_blue, = score_ax.plot([0, 0])
        score_line_olive, = score_ax.plot([0, 0], color='olive')
        score_ax.set_ylim([min_score, max_score])
        score_ax.set_xlim([0, 1])

        plt.title(plot_title)
        plt.xlabel('epoch')
        plt.ylabel('score mean over 5 epoch')
        plt.ion()
        plt.show()
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    time_window = deque(maxlen=10)  # last 10 iter
    best_avg = 13.0
    eval_result = "\n## test result: \n\n"
    for i_episode in range(1, n_episodes + 1):
        state = train_env.reset(train_mode=True)[brain_name].vector_observations[0]
        score = 0
        start = time.time()
        max_reached = False
        while not max_reached:
            action = int(agent.act(state))
            next_state, reward, done, max_reached = helper.unpack_braininfo(brain_name, train_env.step(action))
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        time_window.append(time.time() - start)

        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        print('\rEpisode {}\tAverage Score: {:.2f}\tthis Score: {:.2f}\tAverage Time pr episode {:.2f} seconds'.format(
            i_episode,
            np.mean(
                scores_window),
            score,
            np.mean(
                time_window)),
            end="")
        if plot and i_episode % 5 == 0:
            # score
            # Update axis:
            window = scores[-5:]
            mean = np.mean(window)
            if mean > max_score - buffer:
                max_score = mean + buffer
                score_ax.set_ylim([min_score, max_score])
            if mean < min_score + buffer:
                min_score = mean - buffer
                score_ax.set_ylim([min_score, max_score])
            score_ax.set_xlim([0, len(scores)])
            # PLOT
            helper.plot_score(scores, score_line_blue, score_line_olive)

        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tTime left {:.2f} seconds'.format(i_episode,
                                                                                         np.mean(scores_window),
                                                                                         np.mean(time_window) * (
                                                                                                 n_episodes - i_episode)))
            with open(file, "a+") as f:
                f.write('\tEpisode {}\tAverage Score: {:.2f}\n'.format(i_episode, np.mean(scores_window)))
            if plot:
                plt.savefig(save_img)

        if np.mean(scores_window) >= best_avg:
            # print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}\tTime left {:.2f} seconds'.format(
            #     i_episode,
            #     np.mean(scores_window), np.mean(time_window) * (n_episodes - i_episode)))
            # log_result, current_best = eval(agent, brain_name, train_env, 100, i_episode, save_file, best_avg)
            # eval_result += log_result
            # best_avg = current_best
            debug = 0

        if i_episode % evaluation_interval == 0:
            # Time for evaluation
            log_result, current_best = evaluate(agent, brain_name, train_env, 100, i_episode, save_file, best_avg)
            eval_result += log_result
            best_avg = current_best

    with open(file, "a+") as f:
        f.write(eval_result)
        f.write("\n\nbest score: " + str(max(scores)) + " at eps: " + str(scores.index(max(scores))))
    return scores, best_avg

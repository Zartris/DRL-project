import time
from collections import deque
from pathlib import Path

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from rainbow.agents.rainbow_agent import RainbowAgent
from rainbow.models.ddqn_model import DDQN


def show_plot(scores, Ln):
    # print("plot:")
    mean_scores = []
    for i in range(0, len(scores), 5):
        l = []
        for j in range(5):
            l.append(scores[i + j])
        mean_scores.append(np.mean(l))
    # print(mean_scores)
    Ln.set_ydata(mean_scores)
    Ln.set_xdata(range(0, len(scores), 5))
    plt.pause(0.05)


def dqn(agent, score_file, scheduler=None, save_img="plot.png", save_file='checkpoint.pth', n_episodes=2000000,
        max_t=1000, eps_start=1.0,
        eps_end=0.01,
        eps_decay=0.995, plot=False):
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
        buffer = 25
        min_score = -200
        max_score = min_score + 10
        x_lim = 0
        fig = plt.figure()
        ax = fig.add_subplot(111)
        Ln, = ax.plot([0, 0])
        ax.set_ylim([min_score, max_score])
        ax.set_xlim([0, x_lim])
        plt.xlabel('epoch')
        plt.ylabel('score mean over 5 epoch')
        plt.ion()
        plt.show()

    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    time_window = deque(maxlen=10)  # last 100 scores
    eps = eps_start  # initialize epsilon
    best_avg = 200.0
    env.seed(seed)
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
        if plot and i_episode % 5 == 0:
            # update plot
            window = scores[-5:]
            mean = np.mean(window)
            if mean > max_score - buffer:
                max_score = mean + buffer
                ax.set_ylim([min_score, max_score])
            if mean < min_score + buffer:
                min_score = mean - buffer
                ax.set_ylim([min_score, max_score])
            x_lim += 5
            ax.set_xlim([0, x_lim])
            # threading.Thread(target=show_plot, args=(scores, Ln)).start()
            show_plot(scores, Ln)
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tTime left {:.2f} seconds'.format(i_episode,
                                                                                         np.mean(scores_window),
                                                                                         np.mean(time_window) * (
                                                                                                 n_episodes - i_episode)))
            score_file.write('\tEpisode {}\tAverage Score: {:.2f}\n'.format(i_episode, np.mean(scores_window)))
            if plot:
                plt.savefig(save_img)
        if np.mean(scores_window) >= best_avg:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}\tTime left {:.2f} seconds'.format(
                i_episode - 100,
                np.mean(scores_window), np.mean(time_window) * (n_episodes - i_episode)))
            torch.save(agent.qnetwork_local.state_dict(), str(save_file))
            best_avg = np.mean(scores_window)
    return scores


if __name__ == '__main__':
    env = gym.make("LunarLander-v2")
    # Hyperparameters
    seed = 0
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Agent Hyperparameters
    base_dir = "saved/"
    continues = False
    BUFFER_SIZE = (2 ** 20)
    BATCH_SIZE = 64
    GAMMA = 0.99
    TAU = 1e-3
    LR = 5e-2
    UPDATE_MODEL_EVERY = 2
    UPDATE_TARGET_EVERY = 1000
    use_soft_update = False

    # PER Hyperparameters
    PER_e = 0.01
    PER_a = .6
    PER_b = .3
    PER_bi = 0.01
    PER_aeu = 3
    f = open(base_dir + "scores_tmp.md", "a+")

    # Training
    episodes = 100
    max_t = 1000
    eps_start = 1.0
    eps_end = 0.01
    eps_decay = 0.995
    save_file = base_dir + "rainbow_checkpoint.pth"
    save_image = base_dir + "plot0.png"
    img_path = Path(save_image)
    counter = 0
    while img_path.exists():
        counter += 1
        save_image = base_dir + "plot" + str(counter) + ".png"
        img_path = Path(save_image)
    plot = True

    models = (DDQN(action_size, state_size, seed=seed), DDQN(action_size, state_size, seed=seed))
    agent = RainbowAgent(state_size, action_size, models, seed,
                         continues=continues, BUFFER_SIZE=BUFFER_SIZE, BATCH_SIZE=BATCH_SIZE, GAMMA=GAMMA, TAU=TAU,
                         LR=LR, UPDATE_MODEL_EVERY=UPDATE_MODEL_EVERY, UPDATE_TARGET_EVERY=UPDATE_TARGET_EVERY,
                         use_soft_update=use_soft_update,
                         PER_e=PER_e, PER_a=PER_a, PER_b=PER_b, PER_bi=PER_bi, PER_aeu=PER_aeu)
    dqn(agent, score_file=f, save_img=save_image, save_file=save_file, n_episodes=episodes, max_t=max_t,
        eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay, plot=plot)
    print("Done")

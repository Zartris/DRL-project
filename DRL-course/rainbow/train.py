import time
from collections import deque
from pathlib import Path

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from agents.rainbow_agent import RainbowAgent
from models.models import DDQN


def create_train_info(name, episodes, max_t, eps_start, eps_end, eps_decay):
    t_info = str(name) + "\n"
    t_info += "\tepisodes:" + str(episodes) + "\n"
    t_info += "\tmax_t: " + str(max_t) + "\n"
    t_info += "\teps_start: " + str(eps_start) + "\n"
    t_info += "\teps_end: " + str(eps_end) + "\n"
    t_info += "\teps_decay: " + str(eps_decay) + "\n"
    return t_info


def create_general_info(name, game, seed, state_size, action_size):
    g_info = str(name) + "\n"
    g_info += "\tgame:" + str(game) + "\n"
    g_info += "\tseed: " + str(seed) + "\n"
    g_info += "\tstate_size: " + str(state_size) + "\n"
    g_info += "\taction_size: " + str(action_size) + "\n"
    return g_info


def create_per_info(name, use_per, PER_e, PER_a, PER_b, PER_bi, PER_aeu):
    per_info = str(name) + "\n"
    per_info += "\tuse_per:" + str(use_per) + "\n"
    if use_per:
        per_info += "\tPER_e: " + str(PER_e) + "\n"
        per_info += "\tPER_a: " + str(PER_a) + "\n"
        per_info += "\tPER_b: " + str(PER_b) + "\n"
        per_info += "\tPER_bi: " + str(PER_bi) + "\n"
        per_info += "\tPER_aeu: " + str(PER_aeu) + "\n"
    return per_info


def create_agent_info(name, continues, BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR, UPDATE_MODEL_EVERY, UPDATE_TARGET_EVERY,
                      use_soft_update, priority_method):
    agent_info = str(name) + "\n"
    agent_info += "\tAgent: rainbow\n"
    agent_info += "\tcontinues: " + str(continues) + "\n"
    agent_info += "\tBUFFER_SIZE: " + str(BUFFER_SIZE) + "\n"
    agent_info += "\tBATCH_SIZE: " + str(BATCH_SIZE) + "\n"
    agent_info += "\tGAMMA: " + str(GAMMA) + "\n"
    agent_info += "\tTAU: " + str(TAU) + "\n"
    agent_info += "\tLR: " + str(LR) + "\n"
    agent_info += "\tUPDATE_MODEL_EVERY: " + str(UPDATE_MODEL_EVERY) + "\n"
    agent_info += "\tUPDATE_TARGET_EVERY: " + str(UPDATE_TARGET_EVERY) + "\n"
    agent_info += "\tuse_soft_update: " + str(use_soft_update) + "\n"
    agent_info += "\tpriority_method: " + str(priority_method) + "\n"
    return agent_info


def movingaverage(values, window, mode='same'):
    weights = np.repeat(1.0, window) / window
    sma = np.convolve(values, weights, mode)
    return sma.tolist()


def show_plot(scores, Ln_blue, Ln_olive):
    # print("plot:")
    mean_scores = []
    for i in range(0, len(scores), 5):
        l = []
        for j in range(5):
            l.append(scores[i + j])
        mean_scores.append(np.mean(l))
    # print(mean_scores)
    Ln_blue.set_ydata(mean_scores)
    Ln_blue.set_xdata(range(0, len(scores), 5))
    if len(scores) >= 10:
        yMA = movingaverage(scores, 10)
        Ln_olive.set_ydata(yMA)
        Ln_olive.set_xdata(range(0, len(scores)))
    plt.pause(0.1)


def dqn(agent, file, scheduler=None, save_img="plot.png", save_file='checkpoint.pth', n_episodes=2000000,
        max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995,
        plot=False, plot_title="title"):
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
        fig = plt.figure()
        ax = fig.add_subplot(111)
        Ln_blue, = ax.plot([0, 0])
        Ln_olive, = ax.plot([0, 0], color='olive')
        ax.set_ylim([min_score, max_score])
        ax.set_xlim([0, 1])
        plt.title(plot_title)
        plt.xlabel('epoch')
        plt.ylabel('score mean over 5 epoch')
        plt.ion()
        plt.show()
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    time_window = deque(maxlen=10)  # last 10 iter
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
            ax.set_xlim([0, len(scores)])
            # threading.Thread(target=show_plot, args=(scores, Ln)).start()
            show_plot(scores, Ln_blue, Ln_olive)
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
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}\tTime left {:.2f} seconds'.format(
                i_episode - 100,
                np.mean(scores_window), np.mean(time_window) * (n_episodes - i_episode)))
            torch.save(agent.qnetwork_local.state_dict(), str(save_file))
            best_avg = np.mean(scores_window)
    with open(file, "a+") as f:
        f.write("\n\nbest score: " + str(max(scores)) + " at eps: " + str(scores.index(max(scores))))
    return scores, best_avg


if __name__ == '__main__':
    game = "LunarLander-v2"
    env = gym.make(game)
    # Hyperparameters
    seed = 0
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    general_info = create_general_info("*general info:*", game, seed, state_size, action_size)

    # Agent Hyperparameters

    continues = False
    BUFFER_SIZE = (2 ** 20)
    BATCH_SIZE = 128
    GAMMA = 0.99
    TAU = 1e-4
    LR = 0.001
    UPDATE_MODEL_EVERY = 4
    UPDATE_TARGET_EVERY = 1000
    use_soft_update = False
    priority_method = "reward"
    agent_info = create_agent_info("*agent info:*", continues, BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR,
                                   UPDATE_MODEL_EVERY, UPDATE_TARGET_EVERY, use_soft_update, priority_method)

    # PER Hyperparameters
    use_per = True
    PER_e = 0.01
    PER_a = .6
    PER_b = .4
    PER_bi = 0.001
    PER_aeu = 1
    per_info = create_per_info("*per_info:*", use_per, PER_e, PER_a, PER_b, PER_bi, PER_aeu)

    # Training
    episodes = 200
    max_t = 1000
    eps_start = 1.0
    eps_end = 0.01
    eps_decay = 0.995
    train_info = create_train_info("*train_info:*", episodes, max_t, eps_start, eps_end, eps_decay)

    base_dir = Path("saved", "test0")

    counter = 0
    while base_dir.exists():
        counter += 1
        base_dir = Path("saved", "test" + str(counter))
    base_dir.mkdir(parents=True)
    file = str(Path(base_dir, "model_test.md"))
    save_file = str(Path(base_dir, "rainbow_checkpoint.pth"))
    save_image = str(Path(base_dir, "plot.png"))

    plot = True
    model = DDQN
    title = "model: "
    if model == DDQN:
        title += "Dueling, "
    else:
        title += "Normal, "

    title += "agent: rainbow, "
    if use_per:
        title += "PER-" + priority_method + ", "
    else:
        title += "ER, "

    title += "Update: "
    if use_soft_update:
        title += "soft"
    else:
        title += "hard"

    with open(file, "a+") as f:
        f.write("\n# " + str(title) + "\n\n")
        f.write(general_info + "\n")
        f.write(agent_info + "\n")
        f.write(per_info + "\n")
        f.write(train_info + "\n\n")
        f.write("\n## Test data: \n\n")
    models = (model(state_size, action_size, seed=seed), model(state_size, action_size, seed=seed))
    agent = RainbowAgent(state_size, action_size, models, seed,
                         continues=continues, BUFFER_SIZE=BUFFER_SIZE, BATCH_SIZE=BATCH_SIZE, GAMMA=GAMMA, TAU=TAU,
                         LR=LR, UPDATE_MODEL_EVERY=UPDATE_MODEL_EVERY, UPDATE_TARGET_EVERY=UPDATE_TARGET_EVERY,
                         use_soft_update=use_soft_update, priority_method=priority_method,
                         per=use_per, PER_e=PER_e, PER_a=PER_a, PER_b=PER_b, PER_bi=PER_bi, PER_aeu=PER_aeu)
    dqn(agent, file=file, save_img=save_image, save_file=save_file, n_episodes=episodes, max_t=max_t,
        eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay, plot=plot, plot_title=title)
    print("Done")

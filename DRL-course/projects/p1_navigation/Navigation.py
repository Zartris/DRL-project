import os
import time
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from unityagents import UnityEnvironment

from projects.p1_navigation.agents.rainbow_agent import RainbowAgent
from projects.p1_navigation.models.models import NoisyDDQN, DDQN


def create_model_info(name, std_init):
    m_info = str(name) + "\n"
    m_info += "\tstd_init:" + str(std_init) + "\n"
    return m_info


def create_train_info(name, episodes, evaluation_interval, max_t, eps_start, eps_end, eps_decay):
    t_info = str(name) + "\n"
    t_info += "\tepisodes:" + str(episodes) + "\n"
    t_info += "\tevaluation_interval:" + str(evaluation_interval) + "\n"
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


def create_per_info(name, RB_method, PER_e, PER_a, PER_b, PER_bi, PER_aeu, PER_learn_start, n_step):
    per_info = str(name) + "\n"
    per_info += "\tRB_method:" + str(RB_method) + "\n"
    per_info += "\tPER_e: " + str(PER_e) + "\n"
    per_info += "\tPER_a: " + str(PER_a) + "\n"
    per_info += "\tPER_b: " + str(PER_b) + "\n"
    per_info += "\tPER_bi: " + str(PER_bi) + "\n"
    per_info += "\tPER_aeu: " + str(PER_aeu) + "\n"
    per_info += "\tPER_learn_start " + str(PER_learn_start) + "\n"
    per_info += "\tn_step " + str(n_step) + "\n"
    return per_info


def create_agent_info(name, continues, BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR, opt_eps, UPDATE_MODEL_EVERY,
                      UPDATE_TARGET_EVERY,
                      use_soft_update, priority_method):
    agent_info = str(name) + "\n"
    agent_info += "\tAgent: rainbow\n"
    agent_info += "\tcontinues: " + str(continues) + "\n"
    agent_info += "\tBUFFER_SIZE: " + str(BUFFER_SIZE) + "\n"
    agent_info += "\tBATCH_SIZE: " + str(BATCH_SIZE) + "\n"
    agent_info += "\tGAMMA: " + str(GAMMA) + "\n"
    agent_info += "\tTAU: " + str(TAU) + "\n"
    agent_info += "\tLR: " + str(LR) + "\n"
    agent_info += "\topt_eps: " + str(opt_eps) + "\n"
    agent_info += "\tUPDATE_MODEL_EVERY: " + str(UPDATE_MODEL_EVERY) + "\n"
    agent_info += "\tUPDATE_TARGET_EVERY: " + str(UPDATE_TARGET_EVERY) + "\n"
    agent_info += "\tuse_soft_update: " + str(use_soft_update) + "\n"
    agent_info += "\tpriority_method: " + str(priority_method) + "\n"
    return agent_info


def movingaverage(values, window, mode='same'):
    weights = np.repeat(1.0, window) / window
    sma = np.convolve(values, weights, mode)
    return sma.tolist()


def plot_score(scores, Ln_blue, Ln_olive):
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
    if len(scores) >= 30:
        yMA = movingaverage(scores, 30)
        Ln_olive.set_ydata(yMA)
        Ln_olive.set_xdata(range(0, len(scores)))
    plt.pause(0.1)


def unpack_braininfo(brain_name, all_brain_info):
    brain_info = all_brain_info[brain_name]
    next_state = brain_info.vector_observations[0]
    reward = brain_info.rewards[0]
    done = brain_info.local_done[0]
    max_reached = brain_info.max_reached[0]
    return next_state, reward, done, max_reached


def eval(agent, brain_name, test_env, n_episodes, train_episode="Loaded model", model_save_file=None,
         current_best=10000, set_fast_mode=True):
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    time_window = deque(maxlen=10)  # last 10 iter
    eps = eps_start  # initialize epsilon
    agent.model.eval()
    for i_episode in range(1, n_episodes + 1):
        state = test_env.reset(train_mode=set_fast_mode)[brain_name].vector_observations[0]
        score = 0
        start = time.time()
        max_reached = False
        while not max_reached:
            action = int(agent.act(state, eps))
            next_state, reward, done, max_reached = unpack_braininfo(brain_name, test_env.step(action))
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


def train(agent, brain_name, train_env, file, save_img="plot.png", save_file='checkpoint.pth',
          n_episodes=2000000, evaluation_interval=200, eps_start=1.0, eps_end=0.01, eps_decay=0.995,
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
        buffer = 1
        min_score = 0
        max_score = min_score + buffer
        fig = plt.figure()
        # fig, axs = plt.subplots(2, 1)
        # score_ax = axs[0]
        score_ax = fig.add_subplot(111)
        score_line_blue, = score_ax.plot([0, 0])
        score_line_olive, = score_ax.plot([0, 0], color='olive')
        score_ax.set_ylim([min_score, max_score])
        score_ax.set_xlim([0, 1])

        # loss_ax = axs[1]
        # loss_line_blue, = loss_ax.plot([0, 0])
        # loss_ax.set_ylim([0, 10])
        # loss_ax.set_xlim([0, 1])

        plt.title(plot_title)
        plt.xlabel('epoch')
        plt.ylabel('score mean over 5 epoch')
        plt.ion()
        plt.show()
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    time_window = deque(maxlen=10)  # last 10 iter
    eps = eps_start  # initialize epsilon
    best_avg = 13.0
    eval_result = "\n## test result: \n\n"
    for i_episode in range(1, n_episodes + 1):
        state = train_env.reset(train_mode=True)[brain_name].vector_observations[0]
        score = 0
        start = time.time()
        max_reached = False
        while not max_reached:
            action = int(agent.act(state, eps))
            next_state, reward, done, max_reached = unpack_braininfo(brain_name, train_env.step(action))
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        time_window.append(time.time() - start)

        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
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
            plot_score(scores, score_line_blue, score_line_olive)

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
                i_episode,
                np.mean(scores_window), np.mean(time_window) * (n_episodes - i_episode)))
            # log_result, current_best = eval(agent, brain_name, train_env, 100, i_episode, save_file, best_avg)
            # eval_result += log_result
            # best_avg = current_best

        if i_episode % evaluation_interval == 0:
            # Time for evaluation
            log_result, current_best = eval(agent, brain_name, train_env, 100, i_episode, save_file, best_avg)
            eval_result += log_result
            best_avg = current_best

    with open(file, "a+") as f:
        f.write(eval_result)
        f.write("\n\nbest score: " + str(max(scores)) + " at eps: " + str(scores.index(max(scores))))
    return scores, best_avg


def plot_loss(loss_line_blue, losses):
    if len(losses) >= 5:
        yMA = movingaverage(losses, 5)
        loss_line_blue.set_ydata(yMA)
        loss_line_blue.set_xdata(range(0, len(losses)))


def update_loss_axis(losses, i_episode, loss_ax):
    loss_ax.set_ylim([np.argmin(losses) - 10, np.argmax(losses) + 10])
    loss_ax.set_xlim([0, len(losses)])


if __name__ == '__main__':
    test_agent = False
    cwd = os.getcwd()
    model_to_load = Path(cwd, "saved", "test17", "rainbow_checkpoint.pth")
    # take test_seed before seeding all random variables
    test_seed = np.random.randint(low=1, high=1000)
    # Hyperparameters
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    game = "Banana.exe"
    env = UnityEnvironment(file_name=game, seed=test_seed if test_agent else seed, no_graphics=False)
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)
    general_info = create_general_info("*general info:*", game, seed, state_size, action_size)

    # model parameters:
    std_init = 0.2
    model_info = create_model_info("*model info:*", std_init)

    # Agent Hyperparameters
    continues = False
    BUFFER_SIZE = (2 ** 20)
    BATCH_SIZE = 512
    GAMMA = 0.99
    TAU = 1e-3
    LR = 0.00005
    opt_eps = 1.5e-4  # Adam epsilon
    UPDATE_MODEL_EVERY = 10
    UPDATE_TARGET_EVERY = 8000
    use_soft_update = True
    priority_method = "reward"
    agent_info = create_agent_info("*agent info:*", continues, BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR, opt_eps,
                                   UPDATE_MODEL_EVERY, UPDATE_TARGET_EVERY, use_soft_update, priority_method)

    # PER Hyperparameters
    RB_method = "nstep_per"  # choices: nstep_per, per, replay_buffer
    PER_e = 0.01
    PER_a = 0.6
    PER_b = 0.4
    PER_bi = 0.00001
    PER_aeu = 3
    PER_learn_start = 0
    n_step = 8
    per_info = create_per_info("*per_info:*", RB_method, PER_e, PER_a, PER_b, PER_bi, PER_aeu, PER_learn_start, n_step)

    # Training
    episodes = 2000
    evaluation_interval = 200
    # TODO: REMOVE
    # eps and max_t is not used but is here anyway
    max_t = 1000
    eps_start = 1.0
    eps_end = 0.01
    eps_decay = 0.995
    train_info = create_train_info("*train_info:*", episodes, evaluation_interval, max_t, eps_start, eps_end, eps_decay)

    plot = True
    model = NoisyDDQN
    use_noise = False
    title = "model: "
    if model == DDQN:
        title += "Dueling, "
        models = (model(state_size, action_size, seed=seed), model(state_size, action_size, seed=seed))
    elif model == NoisyDDQN:
        title += "NoisyDueling, "
        use_noise = True
        models = (model(state_size, action_size, seed=seed, std_init=std_init),
                  model(state_size, action_size, seed=seed, std_init=std_init))
    else:
        title += "Normal, "
        models = (model(state_size, action_size, seed=seed), model(state_size, action_size, seed=seed))

    title += "agent: rainbow, "
    if RB_method == "nstep_per":
        title += "NSTEP_PER-" + priority_method + ", "
    elif RB_method == "per":
        title += "PER-" + priority_method + ", "
    else:
        title += "ER, "

    title += "Update: "
    if use_soft_update:
        title += "soft"
    else:
        title += "hard"

    # Test or train
    if test_agent:
        agent = RainbowAgent(state_size, action_size, models, use_noise=use_noise, seed=seed,
                             continues=continues, BUFFER_SIZE=BUFFER_SIZE, BATCH_SIZE=BATCH_SIZE, GAMMA=GAMMA, TAU=TAU,
                             LR=LR, UPDATE_MODEL_EVERY=UPDATE_MODEL_EVERY, UPDATE_TARGET_EVERY=UPDATE_TARGET_EVERY,
                             use_soft_update=use_soft_update, priority_method=priority_method,
                             RB_method=RB_method, PER_e=PER_e, PER_a=PER_a, PER_b=PER_b, PER_bi=PER_bi, PER_aeu=PER_aeu,
                             PER_learn_start=PER_learn_start, n_step=n_step)
        state_dict = torch.load(str(model_to_load))
        agent.model.load_state_dict(state_dict)
        eval(agent, brain_name, env, 100, set_fast_mode=True)
    else:
        base_dir = Path("saved", "test0")

        counter = 0
        while base_dir.exists():
            counter += 1
            base_dir = Path("saved", "test" + str(counter))
        base_dir.mkdir(parents=True)
        file = str(Path(base_dir, "model_test.md"))
        save_file = str(Path(base_dir, "rainbow_checkpoint.pth"))
        save_image = str(Path(base_dir, "plot.png"))

        with open(file, "a+") as f:
            f.write("\n# " + str(title) + "\n\n")
            f.write(general_info + "\n")
            f.write(model_info + "\n")
            f.write(agent_info + "\n")
            f.write(per_info + "\n")
            f.write(train_info + "\n\n")
            f.write("\n## train data: \n\n")
        agent = RainbowAgent(state_size, action_size, models, use_noise=use_noise, seed=seed,
                             continues=continues, BUFFER_SIZE=BUFFER_SIZE, BATCH_SIZE=BATCH_SIZE, GAMMA=GAMMA, TAU=TAU,
                             LR=LR, UPDATE_MODEL_EVERY=UPDATE_MODEL_EVERY, UPDATE_TARGET_EVERY=UPDATE_TARGET_EVERY,
                             use_soft_update=use_soft_update, priority_method=priority_method,
                             RB_method=RB_method, PER_e=PER_e, PER_a=PER_a, PER_b=PER_b, PER_bi=PER_bi, PER_aeu=PER_aeu,
                             PER_learn_start=PER_learn_start, n_step=n_step)
        train(agent, brain_name, env, file=file, save_img=save_image, save_file=save_file, n_episodes=episodes,
              evaluation_interval=evaluation_interval, eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay,
              plot=plot,
              plot_title=title)
        if Path(save_file).exists():
            agent.model.state_dict(torch.load(save_file))
            agent.model_target.state_dict(torch.load(save_file))
        # eval(agent, brain_name, env, file, 100)
        print("Done")

import matplotlib.pyplot as plt
import numpy as np


def movingaverage(values, window, mode='same'):
    weights = np.repeat(1.0, window) / window
    sma = np.convolve(values, weights, mode)
    return sma.tolist()


def plot_score(scores, Ln_blue, Ln_olive):
    # print("plot:")
    mean_scores = []
    mean_scores.append(0)
    for i in range(0, len(scores), 5):
        l = []
        for j in range(5):
            l.append(scores[i + j])
        mean_scores.append(np.mean(l))
    # print(mean_scores)
    Ln_blue.set_ydata(mean_scores)
    Ln_blue.set_xdata(range(0, len(scores)+1, 5))
    if len(scores) >= 30:
        yMA = movingaverage(scores, 30)
        Ln_olive.set_ydata(yMA)
        Ln_olive.set_xdata(range(0, len(scores)))
    plt.pause(0.1)


def plot_loss(loss_line_blue, losses):
    if len(losses) >= 5:
        yMA = movingaverage(losses, 5)
        loss_line_blue.set_ydata(yMA)
        loss_line_blue.set_xdata(range(0, len(losses)))


def update_loss_axis(losses, i_episode, loss_ax):
    loss_ax.set_ylim([np.argmin(losses) - 10, np.argmax(losses) + 10])
    loss_ax.set_xlim([0, len(losses)])


def unpack_braininfo(brain_name, all_brain_info):
    brain_info = all_brain_info[brain_name]
    next_state = brain_info.vector_observations[0]
    reward = brain_info.rewards[0]
    done = brain_info.local_done[0]
    max_reached = brain_info.max_reached[0]
    return next_state, reward, done, max_reached

import random

import matplotlib.pyplot as plt
import numpy as np


class Test:
    def __init__(self, seed):
        np.random.seed(seed)

    def run_test(self):
        l = []
        l2 = []
        for i in range(10):
            l.append(np.random.random())
            l2.append(random.random())
        print(l)
        print(l2)


if __name__ == '__main__':
    test = Test(0)
    test2 = Test(1)
    test.run_test()
    test2.run_test()

    dat = [0, 1]
    fig, axs = plt.subplots(2, 1)
    score_ax = axs[0]
    loss_ax = axs[1]
    Ln, = score_ax.plot(dat)
    score_ax.set_ylim([0, 10])
    score_ax.set_xlim([0, 20])
    loss_line, = loss_ax.plot(dat)
    loss_ax.set_xlim([0, 18 * 5])
    loss_ax.set_ylim([0, 10])
    plt.ion()
    plt.show()
    dat2 = []
    for i in range(18):
        dat.append(random.uniform(0, 10))
        Ln.set_ydata(dat)
        Ln.set_xdata(range(len(dat)))
        for j in range(5):
            dat2.append(random.uniform(0, 10))
            loss_line.set_ydata(dat2)
            loss_line.set_xdata(range(len(dat2)))
        plt.pause(1)

        print('done with loop')
    plt.savefig('foo.png')

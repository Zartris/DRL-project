import numpy as np
import matplotlib.pyplot as plt


class Test:
    def __init__(self, seed):
        np.random.seed(seed)

    def run_test(self):
        l = []
        for i in range(10):
            l.append(np.random.random())
        print(l)


if __name__ == '__main__':
    test = Test(0)
    test2 = Test(1)
    test.run_test()
    test2.run_test()

    import pylab
    import time
    import random
    import matplotlib.pyplot as plt

    dat = [0, 1]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    Ln, = ax.plot(dat)
    ax.set_xlim([0, 20])
    plt.ion()
    plt.show()
    for i in range(18):
        dat.append(random.uniform(0, 10))
        Ln.set_ydata(dat)
        Ln.set_xdata(range(len(dat)))
        plt.pause(1)

        print ('done with loop')
    plt.savefig('foo.png')
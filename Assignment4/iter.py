import numpy as np
import matplotlib.pyplot as plt
from E import *


def method_iter(n, w, n_restart, nh_size):
    local_minima = []
    plt.figure()
    for t in range(n_restart):

        # Question: do I need to wait for convergence? It is a whole, exhaustive search right?

        x = 2 * (np.random.uniform(0, 1, n) > 0.5).astype(int) - 1
        E1 = [meanE(x, w)]

        # Question: what is this flag?

        flag = 1
        while flag == 1:
            flag = 0
            if nh_size == 1:
                # choose new x by flipping one bit i
                for i in range(len(x)):
                    x_new = x.copy()
                    x_new[i] = -x[i]
                    if meanE(x_new, w) < meanE(x, w):
                        x = x_new
                        flag = 1
                    E1.append(meanE(x, w))

            elif nh_size == 2:
                # choose new x by flipping bits i,j: DRAW
                for i in range(len(x)):
                    x_new_i = x.copy()
                    x_new_i[i] = -x[i]
                    for j in range(len(x)):
                        x_new_j = x_new_i.copy()
                        x_new_j[j] = -x_new_i[j]
                        if meanE(x_new_j, w) < meanE(x_new_i, w):
                            x_new_i = x_new_j
                    if meanE(x_new_i, w) < meanE(x, w):
                        x = x_new_i
            E1.append(meanE(x, w))

        local_minima.append(E1[len(E1) - 1])
        plt.subplot(5, 2, t+1)
        plt.plot(np.linspace(0, len(E1), len(E1)), E1, label="mean E")
        plt.title("Iterative Improvement on Ising Model, Emin = %f" % local_minima[t])
        plt.xlabel("iterations")
        plt.ylabel("E(x)")
    plt.show()
    return local_minima


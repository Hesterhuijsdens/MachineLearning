import numpy as np
import random


def E(x, w):
    return -0.5 * x * w * x.T


def meanE(x, w):
    return np.sum((E(x, w))) / len(x)


def flip_prob(x, x_new, w, b):
    """ prob x|w """
    if meanE(x_new, w) < meanE(x, w):
        return 1
    else:
        return np.exp(b * (meanE(x_new, w) - meanE(x, w)))


def state_transition(x, n, w, beta):
    index = random.randint(0, n - 1)
    x_new = x.copy()
    x_new[index] = -x[index]

    # compute flip probability
    p_flip = flip_prob(x, x_new, w, beta)

    # determine new binary state
    if np.random.rand() < 1. / p_flip:
        x[:] = x_new
    return meanE(x, w)


# def metropolis_hastings(p, T1):
#     samples = np.zeros((iter, T1))
#
#     for i in range(T1):
#         x_star, y_star = np.array([x, y]) + np.random.normal(size=2)
#         if np.random.rand() < p(x_star, y_star) / p(x, y):
#             x, y = x_star, y_star
#         samples[i] = np.array([x, y])
#
#     return samples

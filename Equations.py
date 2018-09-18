import numpy as np
from math import log


# fully connected layer
def linear(x, w):
    return np.dot(w, x)


# sigmoid
def sigmoid(a):
    return 1 / (1 + np.exp(-a.astype(float)))


# forward propagation
# input x = [P N] (data matrix of P inputs for N examples), w = [1 P] (weight matrix)
# output = [1 N] for N examples
def forward(x, w):
    return sigmoid(linear(x, w))


# loss function
def cost(y, t):

    # [item += 0.0001 for image in y for item in image if item == 0]

    N = np.shape(t)[0]

    for image in y:
        for item in image:
            if item == 0:
                item += 0.0001

    print np.shape(y)
    return -1 / N * np.sum(t * np.log(y) + (1 - np.transpose(t)) * np.log(1 - y))

    # sum = 0
    # for n in range(1, N):
    #     if(y[0, n] != 0 and (1 - y[0, n] != 0)):
    #         sum += t[n] * log(y[0, n]) + (1 - t[n]) * log(1 - y[0, n])
    #     elif y[0, n] == 0:
    #         sum += t[n] * -inf + (1 - t[n]) * log(1 - y[0, n])
    #     else:
    #         sum += t[n] * log(y[0, n]) + (1 - t[n]) * -inf
    #
    # return -1/N * sum

    # return (-1/N * np.sum(t * np.log(y) + (1 - t) * np.log(1 - y)))
    # print y, t, N
    # print -1/N * np.sum(t * np.log(y) + (1 - t) * np.log(1 - y))


# the gradient; backward propagation computes the backward pass for a one-layer network
# with sigmoid units and loss
def backward(x, y, t):
    return np.dot((y - np.transpose(t)), x)


def gradient_e_decay(N, y, t, x, decay, w):    # with weight decay
    sum = 0
    for n in range(1, N):
        sum += (y[0, n] - t[n]) * x[n, :] + (decay/n)*w
    return (1/N) * sum


def hessian(N, x, y, decay, d):   # with weight decay
    H = np.ones((d, d))
    for i in range(d):
        for j in range(d):
            sum = 0
            for n in range(N):
                sum += x[n, i] * y[0, n] * (1 - y[0, n]) * x[n, j]
                if i == j and n > 0:
                    sum += decay/n
            H[i, j] = (1/N) * sum
    return H


def classification_error(y, t):
    mistakes = 0
    for i in range(np.shape(y)[1]):
        if (y[0, i] > 0.5 and t == 0) or (y[0, i] < 0.5 and t == 1):
            mistakes += 1
    return mistakes / np.shape(y)[1]



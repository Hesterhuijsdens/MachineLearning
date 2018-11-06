import numpy as np


def initialize_weights(n_in, n_out):
    """
    Draw weights uniformly from the rule of thumb range
    (instead of from Gaussian distribution)
    """
    r = np.sqrt(6) / np.sqrt(n_out + n_in)
    return np.random.uniform(-r, r, [n_out, n_in])


def linear(X, W):
    return np.dot(W, X)


def sigmoid(A):
    return 1 / (1 + np.exp(-A))


def cross_entropy(Y, T):

    # nonzero only
    for y in Y:
        for item in y:
            if item == 0:
                y += 0.0001
    m, n = Y.shape
    return -np.sum(T * np.log(Y)) / (m * n)


def softmax(Z):
    # for numerical stability
    Z -= np.max(Z, axis=0)
    return np.exp(Z) / np.sum(np.exp(Z), axis=0)


def forward(X, W1, W2):
    H = sigmoid(linear(X, W1))
    Y = softmax(linear(H, W2))
    return H, Y


def backward(X, H, Y, W2, T):

    # second layer
    dW2 = np.dot((Y - T), np.transpose(H))

    # first layer
    dot1 = np.dot(np.transpose(Y - T), W2)
    multi1 = dot1 * np.transpose(H)
    multi2 = multi1 * np.transpose(1 - H)
    dW1 = np.dot(np.transpose(multi2), X)
    return dW1, dW2
























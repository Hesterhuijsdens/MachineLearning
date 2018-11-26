import numpy as np
import math
import time


# fully connected layer
def linear(x, w):
    return np.dot(w, x)


# sigmoid
def sigmoid(a):
    return 1.0 / (1.0 + np.exp(-a.astype(float)))


# forward propagation
# input x = [P N] (data matrix of P inputs for N examples), w = [1 P] (weight matrix)
# output = [1 N] for N examples
def forward(x, w):
    return sigmoid(linear(x, w))


# loss function E(w):
def cost(y, t):
    y[y < 0.001] = 0.001
    y[y > 0.999] = 0.999
    N = np.shape(t)[0]
    return (-1.0 / N) * (np.sum((t * np.log(y[0]) + (1 - np.transpose(t)) * np.log(1 - y[0]))))


# weight decay: error function with regularization
# def cost_reg(y, t, reg, w):
#     return cost(y, t) + (reg/2) * np.dot(w, np.transpose(w))


#  loss function with weight decay
def cost_decay(y, t, decay, w):
    y[y < 0.001] = 0.001
    y[y > 0.999] = 0.999
    N = np.shape(t)[0]
    cost_without = np.sum(t * np.log(y[0]) + (1 - np.transpose(t)) * np.log(1 - y[0]))
    sum = 0
    for n in range(1, N + 1):
        sum += (decay/(2*n)) * np.dot(w[0, 0:n+1], w[0, 0:n+1])
    return (-1.0 / N) * (cost_without + sum)


# the gradient; backward propagation computes the backward pass for a one-layer network
# with sigmoid units and loss
def backward(x, y, t):
    N = np.shape(t)[0]
    return (1.0/N) * np.dot((y[0] - np.transpose(t)), x)


# the gradient of E with weight decay:
def gradient_e_decay(y, t, x, decay, w):
    N = np.shape(x)[0]
    gradient = np.zeros(np.shape(w)[1])
    for i in range(np.shape(w)[1]):
        formula_inside = (y - t) * x[:, i] + np.reshape((decay / np.arange(1, N + 1)) * w[0, i], (1, N))
        gradient[i] = 1.0 / N * np.sum(formula_inside)
    return gradient


# computes the Hessian with weight decay:
def hessian(x, y, decay):
    N = np.shape(x)[0]
    d = np.shape(x)[1]
    formula_inside = 0
    for n in range(N):
        formula_inside += np.matmul(np.transpose(np.reshape(x[n, :], (1, d)) * y[0, n] * (1 - y[0, n])),
                                    np.reshape(x[n, :], (1, d)))
    return (1.0 / N) * (formula_inside + np.sum(decay / np.arange(1, N + 1)) * np.identity(d))


# computes the percentage of misclassified patterns
def classification_error(y, t):
    mistakes = 0
    for i in range(np.shape(y)[1]):
        if (y[0, i] > 0.5 and t[i] == 0) or (y[0, i] < 0.5 and t[i] == 1):
            mistakes += 1
    return (float(mistakes) / np.shape(y)[1]) * 100.0


# to compute the step size for line search:
def golden_section_search(a, b, w, x, t, direction, tolerance=1e-5):
    ratio = (math.sqrt(5) + 1) / 2
    c = b - (b - a) / ratio
    d = a + (b - a) / ratio

    while abs(c - d) > tolerance:
        # cost(y, t)
        # error_function(w, x, t)
        y_c = forward(np.transpose(x), w + direction * c)
        y_d = forward(np.transpose(x), w + direction * d)

        if cost(y_c, t) < cost(y_d, t):
            b = d
        else:
            a = c

        c = b - (b - a) / ratio
        d = a + (b - a) / ratio

    return (b + a) / 2


def training(n_epochs, x, t, x_val, t_val, w, dw, eta=1, decay=0.1, alpha=0.1, momentum=0, weightdecay=0):
    start = time.time()
    train_loss = np.zeros(n_epochs)
    val_loss = np.zeros(n_epochs)
    dW = dw
    y_save = 0
    y_val = 0

    for epoch in range(n_epochs):
        y = forward(np.transpose(x), w)

        if weightdecay == 0:
            gradE = backward(x, y, t)
        else:
            gradE = gradient_e_decay(y, t, x, decay, w)

        if epoch == n_epochs - 1:
            y_save = y

        if momentum == 0:
            dW = -eta * gradE
            w = w + dW
        else:
            dW = -eta * gradE + alpha * dW
            w = w + dW

        # compute loss
        if weightdecay == 0:
            train_loss[epoch] = cost(y, t)
        else:
            train_loss[epoch] = cost_decay(y, t, decay, w)

        # validation
        y_val = forward(np.transpose(x_val), w)

        if weightdecay == 0:
            val_loss[epoch] = cost(y_val, t_val)
        else:
            val_loss[epoch] = cost_decay(y_val, t_val, decay, w)

    end = time.time()
    return train_loss, y_save, val_loss, y_val, w, end, start


def testing(x, w, t):
    y = forward(np.transpose(x), w)
    accuracy = 0
    for i in range(len(t)):
        if int(round(y[0, i])) == t[i]:
            accuracy += 1
    return float(accuracy) / len(t) * 100.0


# function to compute beta using Polak-Ribiere rule:
def polak_ribiere(old_decay, new_decay):
    magnitude = np.linalg.norm(old_decay)
    beta = ((new_decay - old_decay) * new_decay) / np.power(magnitude, 2)
    return beta




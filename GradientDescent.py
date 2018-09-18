from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
from PreprocessData import load
# from math import e


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
def cost(y, t, N):
    # [y += 0.0001 for im in y for item in im if item == 0]

    # for image in y:
    #     # for item in image:
    #     if image == 0:
    #         y += 0.0001

    return -1/N * np.sum(t * np.log(y) + (1 - np.transpose(t)) * np.log(1 - y))

# the gradient; backward propagation computes the backward pass for a one-layer network
# with sigmoid units and loss
def backward(x, y, t):
    return np.dot((y - np.transpose(t)), x)


# load data
x37_train, t37_train = load()
x37_train = x37_train[:500]
t37_train = t37_train[:500]
x37_val = x37_train[501:600]
t37_val = t37_train[501:600]

# initialize weights
w = np.random.randn(1, np.shape(x37_train)[1])

# total number of patterns
N = np.shape(x37_train)[0]

# training
n_epochs = 70
eta = 0.01
xaxis = []

# arrays for saving losses
train_loss = np.zeros(n_epochs)
val_loss = np.zeros(n_epochs)
for epoch in range(n_epochs):

    # forward propagation
    y37_train = forward(np.transpose(x37_train), w)

    # backward propagation
    dW = backward(x37_train, y37_train, t37_train)

    # weight update
    w = w - (eta * dW)

    # compute loss
    train_loss[epoch] = cost(y37_train, t37_train, N)
    xaxis.append(epoch)

    y37_val = forward(np.transpose(x37_val), w)
    val_loss[epoch] = cost(y37_val, t37_val, N)


plt.figure(0)
plt.plot(xaxis, train_loss)
plt.plot(xaxis, val_loss)
plt.legend(["training set", "validation set"])
plt.show()




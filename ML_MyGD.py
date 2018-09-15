from mnist import MNIST
import numpy as np

# load data, select 3 and 7
data = MNIST('MachineLearning/data')
x_train, y_train = data.load_training()
x3_train = [x_train[i] for i in range(0, len(y_train)) if y_train[i] == 3]
x7_train = [x_train[i] for i in range(0, len(y_train)) if y_train[i] == 7]
x37_train = x3_train + x7_train

x_test, y_test = data.load_testing()


# fully connected layer
def linear(x, w):
    return np.dot(w, x)


# sigmoid
def sigmoid(a):
    return 1 / (1 + np.exp(-a))


# forward propagation
# input x = [P N] (data matrix of P inputs for N examples), w = [1 P] (weight matrix)
# output = [1 N] for N examples
def forward(x, w):
    return sigmoid(linear(x, w))


# log likelihood
def loss(y, t, N):
    for image in y:
        for item in image:
            if item == 0:
                y += 0.0001
    return -1/N * np.sum(t * np.log(y) + (1 - t) * np.log(1 - y))


# the gradient; backward propagation computes the backward pass for a one-layer network
# with sigmoid units and loss
def backward(x, y, t):
    return np.dot((y - t), np.transpose(x))


# Hessian
# def hessian(x, y, N):
#     return -1/N * np.sum()


# initialize weights
w = np.random.randn(1, np.shape(x37_train[1]))

# total number of patterns
N = np.shape(x37_train[0])

# training
epochs = 10
for epoch in xrange(epochs):

    # forward prop
    y37_train = forward(np.transpose(x37_train), w)

    # backprop
    dW = backward(x37_train, y37_train, t_train)

    # learning rate
    eta = 0.001

    # weight update
    w = w - (eta * dW)

    # compute loss
    loss = loss(y37_train, t_train, N)








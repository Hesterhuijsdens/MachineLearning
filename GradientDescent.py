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
    # print "do forward"
    return sigmoid(linear(x, w))


# loss function
def cost(y, t, N):


    for image in y:
        for item in image:
            if item == 0:
                y += 0.0001
    a = -t * np.log(y)
    # print a

    # print np.shape(t), np.shape(y)

    b = (1 - t) * np.log(1 - y)
    # print "a", a[0][:10]
    # print "b", b[0][:10]
    # print "a-b", (a - b)[0][:10]
    print "a", a
    print "b", b
    print "a-b", (a - b)

    # print np.shape(a), np.shape(b)
    # print np.shape((a - b))

    bb = a - b
    # print bb

    c = np.sum(bb)
    print c
    # print N
    # print type(N)
    # print 1/N * c
    return 1/N * c

    # return -1/N * np.sum(t * np.log(y) + (1 - t) * np.log(1 - y))


# the gradient; backward propagation computes the backward pass for a one-layer network
# with sigmoid units and loss
def backward(x, y, t):
    # print "do backward"

    return np.dot((y - np.transpose(t)), x)


# Hessian
# def hessian(x, y, N):
#     return -1/N * np.sum()


# load data
x37_train, t37_train = load()
x37_train = x37_train[:4]
t37_train = t37_train[:4]

# initialize weights
w = np.random.randn(1, np.shape(x37_train)[1])

# total number of patterns
N = np.shape(x37_train)[0]

# training
epochs = 50
eta = 0.001
losses = []
xaxis = []
for epoch in range(epochs):
    print('hoi')
    # forward propagation
    y37_train = forward(np.transpose(x37_train), w)

    # backward propagation
    dW = backward(x37_train, y37_train, t37_train)

    # weight update
    w = w - (eta * dW)

    # compute loss
    losses.append(cost(y37_train, t37_train, N))
    xaxis.append(epoch)


plt.figure(0)
plt.plot(xaxis, losses)
plt.show()




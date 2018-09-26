import numpy as np


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
    y[y < 0.001] = 0.001
    y[y > 0.999] = 0.999
    N = np.shape(t)[0]
    return (-1.0 / N) * np.sum(t * np.log(y) + (1 - np.transpose(t)) * np.log(1 - y))


#  loss function with weight decay
def cost_decay(y, t, decay, w):
    N = np.shape(t)[0]
    y[y < 0.001] = 0.001
    y[y > 0.999] = 0.999
    cost = (-1.0 / N) * np.sum(t * np.log(y) + (1 - np.transpose(t)) * np.log(1 - y))
    sum = 0
    for n in range(1, N):
        sum += (decay/(2*n)) * np.dot(w[0, 0:n], w[0, 0:n])
    return cost - (1.0/N) * sum


# the gradient; backward propagation computes the backward pass for a one-layer network
# with sigmoid units and loss
def backward(x, y, t):
    return np.dot((y - np.transpose(t)), x)


# the gradient of E with weight decay:
def gradient_e_decay(y, t, x, decay, w):
    N = np.shape(x)[0]
    sum = 0
    for n in range(1, N):
        sum += (decay/n) * w
    return (1.0/N) * np.matmul(y - np.transpose(t), x) + np.mean(sum, 0)


# computes the Hessian with weight decay:
def hessian(x, y, decay, d):
    N = np.shape(x)[0]
    H = np.ones((d, d))
    for i in range(d):
        for j in range(d):
            ans = np.multiply(np.multiply(np.multiply(x[:,i], y), 1-y), x[:,j])
            if i == j:
                for n in range(1, N): # weight decay
                    ans += (decay/n)
            H[i, j] = np.sum(ans)
    return (1.0/N) * H


# Computes the percentage of misclassified patterns (for NewtonMethod):
def classification_error(y, t):
    mistakes = 0
    for i in range(np.shape(y)[1]):
        if (y[0, i] > 0.5 and t[i] == 0) or (y[0, i] < 0.5 and t[i] == 1):
            mistakes += 1
    return float(mistakes) / np.shape(y)[1]



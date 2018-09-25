import matplotlib.pyplot as plt
from PreprocessData import load
from Equations import *

# avoid overflow warnings
np.seterr(all="ignore")

# load data
x37_training, t37_training = load()
x37_train = x37_training[:500]
t37_train = t37_training[:500]
x37_val = x37_training[501:600]
t37_val = t37_training[501:600]

# initialize weights
w = np.random.randn(1, np.shape(x37_train)[1])

# total number of patterns
N = np.shape(x37_train)[0]

# training
n_epochs = 10000
eta = 0.001
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
    train_loss[epoch] = cost(y37_train, t37_train)
    xaxis.append(epoch)

    y37_val = forward(np.transpose(x37_val), w)
    val_loss[epoch] = cost(y37_val, t37_val)
print train_loss[9999], val_loss[9999]


plt.figure(0)
plt.plot(xaxis, train_loss)
plt.plot(xaxis, val_loss)
plt.legend(["training set", "validation set"])
plt.show()




import matplotlib.pyplot as plt

from Assignment2.Equations_mlp import *
from PreprocessData import load

# avoid overflow warnings
np.seterr(all="ignore")


# load data (N=12396L)
x37_training, t37_training = load()
# lb = 499
# ub = 599

lb = 9999
ub = np.shape(x37_training)[0] - 1

x37_train = x37_training[:lb]
t37_train = t37_training[:lb]
x37_val = x37_training[lb+1:ub]
t37_val = t37_training[lb+1:ub]

# hyper parameters
n_hidden = 30
n_output = 10
n_epochs = 10
eta = 0.0001

# initialize weights
W1 = initialize_weights(x37_train.shape[1], n_hidden)
W2 = initialize_weights(n_hidden, n_output)

train_loss = np.zeros(n_epochs)
val_loss = np.zeros(n_epochs)
for epoch in xrange(n_epochs):

    # forward propagation
    h37_train, y37_train = forward(x37_train.T, W1, W2)

    # backward propagation
    dW1, dW2 = backward(x37_train, h37_train, y37_train, W2, t37_train)

    # weight update
    W2 -= eta * dW2
    W1 -= eta * dW1

    # compute loss
    loss = cross_entropy(y37_train, t37_train)
    train_loss[epoch] = loss

    # validation
    h37_val, y37_val = forward(x37_val.T, W1, W2)
    loss_val = cross_entropy(y37_val, t37_val)
    val_loss[epoch] = loss_val

plt.plot(range(len(train_loss)), train_loss, label='train loss')
plt.plot(range(len(val_loss)), val_loss, label='test loss')
plt.legend()
plt.title('Training and validation loss over %i epochs' %n_epochs)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()



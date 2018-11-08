import numpy as np
import matplotlib.pyplot as plt
from PreprocessData import load37
from Equations import *
import time

# load train and test data:
x, t = load37(version="train")
x = x[:300]
t = t[:300]
x_test, t_test = load37(version="test")
x_test = x_test[:100]
t_test = t_test[:100]

# store dimensions of data:
N = np.shape(x)[0]
d = np.shape(x)[1]

# set parameters:
decay = 0.1
epochs = 200 # 5000
eta = 1
alpha = 0.1

# initialize loss arrays
train_loss, train_loss_m, val_loss, val_loss_m, train_loss_wd, val_loss_wd, train_loss_wdm, val_loss_wdm = \
    (np.zeros(epochs) for list in range(8))
xaxis = []

# initialize weights
w, wm, wwd, wwdm = (np.random.randn(1, d) for weights in range(4))

# initialize gradients
dW, dWm, dWwd, dWwdm = (np.random.randn(1) for i in range(4))

for epoch in range(epochs):
    print "Epoch: ", epoch

    # get minibatch:
    random_indices = np.random.uniform(0, N, int(0.01*N)).astype(int)
    batch_x = x[random_indices]
    batch_t = np.reshape(np.reshape(t, (N, 1))[random_indices], (int(0.01*N),))

    # forward propagation:
    y = forward(np.transpose(batch_x), w)
    ym = forward(np.transpose(batch_x), wm)
    ywd = forward(np.transpose(batch_x), wwd)
    ywdm = forward(np.transpose(batch_x), wwdm)

    # backward propagation:
    gradE = backward(batch_x, y, batch_t)
    gradE_m = backward(batch_x, ym, batch_t)
    gradE_wd = gradient_e_decay(ywd, batch_t, batch_x, decay, w)
    gradE_wdm = gradient_e_decay(ywd, batch_t, batch_x, decay, w)

    # weight update (regular)
    dW = -eta * gradE
    w = w + dW

    # weight update with momentum
    dWm = -eta * gradE_m + alpha * dWm
    wm = wm + dWm

    # weight update with weight decay
    dWwd = -eta * gradE_wd
    wwd = wwd + dWwd

    # weight update with momentum and weight decay
    dWwdm = -eta * gradE_wdm + alpha * dWwdm
    wwdm = wwdm + dWwdm

    # compute loss
    train_loss[epoch] = cost(y, batch_t)
    train_loss_m[epoch] = cost(ym, batch_t)
    train_loss_wd[epoch] = cost_decay(ywd, batch_t, decay, wwd)
    train_loss_wdm[epoch] = cost_decay(ywdm, batch_t, decay, wwdm)
    xaxis.append(epoch)

    # # val on w
    # y37_val = forward(np.transpose(x37_val), w)
    # val_loss[epoch] = cost(y37_val, t37_val)
    #
    # # val on wm
    # y37_val_m = forward(np.transpose(x37_val), wm)
    # val_loss_m[epoch] = cost(y37_val_m, t37_val)
    #
    # # val on wwd
    # y37_val_wd = forward(np.transpose(x37_val), wwd)
    # val_loss_wd[epoch] = cost_decay(y37_val_wd, t37_val, decay=alpha, w=wwd)
    #
    # # val on wwdm
    # y37_val_wdm = forward(np.transpose(x37_val), wwdm)
    # val_loss_wdm[epoch] = cost_decay(y37_val_wdm, t37_val, decay=alpha, w=wwdm)

# SGD
plt.subplot(2, 2, 1)
plt.plot(xaxis, train_loss)
plt.plot(xaxis, val_loss)
plt.legend(["train", "test"])
plt.title("Stochastic gradient descent (eta=%1.3f)" %eta)
plt.xlabel("N")
plt.ylabel("loss")

# SGD with momentum
plt.subplot(2, 2, 2)
plt.plot(xaxis, train_loss_m)
plt.plot(xaxis, val_loss_m)
plt.legend(["train", "test"])
plt.title("Momentum (alpha=%1.3f)" %alpha)
plt.xlabel("N")
plt.ylabel("loss")

# SGD with weight decay
plt.subplot(2, 2, 3)
plt.plot(xaxis, train_loss_wd)
plt.plot(xaxis, val_loss_wd)
plt.legend(["train", "test"])
plt.title("Weight decay")
plt.xlabel("N")
plt.ylabel("loss")

# SGD with momentum and weight decay
plt.subplot(2, 2, 4)
plt.plot(xaxis, train_loss_wdm)
plt.plot(xaxis, val_loss_wdm)
plt.legend(["train", "test"])
plt.title("Weight decay + momentum (alpha=%1.3f)" %alpha)
plt.xlabel("N")
plt.ylabel("loss")
plt.show()




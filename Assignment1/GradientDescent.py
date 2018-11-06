import matplotlib.pyplot as plt
from PreprocessData import load
from Equations import *

# avoid overflow warnings
np.seterr(all="ignore")

# load data (N=12396L)
x37_training, t37_training = load()

# bounds train and validation set
lb = 9999
ub = np.shape(x37_training)[0] - 1
# lb = 999
# ub = 1299
x37_train = x37_training[:lb]
t37_train = t37_training[:lb]
x37_val = x37_training[lb+1:ub]
t37_val = t37_training[lb+1:ub]

# total number of patterns
N = np.shape(x37_train)

# hyper parameters
n_epochs = 1000
eta = 20
alpha = 0.1

# arrays for saving losses and predictions
train_loss, train_loss_m, val_loss, val_loss_m, train_loss_wd, val_loss_wd, train_loss_wdm, val_loss_wdm = \
    (np.zeros(n_epochs) for list in range(8))
y37, y37_m, y37_wd, y37_wdm = (0 for i in range(4))

# initialize weights
w, wm, wwd, wwdm = (np.random.randn(1, np.shape(x37_train)[1]) for weights in range(4))

# initialize gradients
dW, dWm, dWwd, dWwdm = (0 for i in range(4))

xaxis = []
for epoch in range(n_epochs):
    # forward propagation
    y37_train = forward(np.transpose(x37_train), w)
    y37_train_m = forward(np.transpose(x37_train), wm)
    y37_train_wd = forward(np.transpose(x37_train), wwd)
    y37_train_wdm = forward(np.transpose(x37_train), wwdm)

    # save predictions last iteration
    if epoch == n_epochs-1:
        y37 = y37_train
        y37_m = y37_train_m
        y37_wd = y37_train_wd
        y37_wdm = y37_train_wdm

    # backward propagation
    gradE = backward(x37_train, y37_train, t37_train)
    gradE_m = backward(x37_train, y37_train_m, t37_train)
    gradE_wd = backward(x37_train, y37_train_wd, t37_train)
    gradE_wdm = backward(x37_train, y37_train_wdm, t37_train)

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
    train_loss[epoch] = cost(y37_train, t37_train)
    train_loss_m[epoch] = cost(y37_train_m, t37_train)
    train_loss_wd[epoch] = cost_decay(y37_train_wd, t37_train, decay=0.8, w=wwd)
    train_loss_wdm[epoch] = cost_decay(y37_train_wdm, t37_train, decay=0.8, w=wwdm)
    xaxis.append(epoch)

    # val on w
    y37_val = forward(np.transpose(x37_val), w)
    val_loss[epoch] = cost(y37_val, t37_val)

    # val on wm
    y37_val_m = forward(np.transpose(x37_val), wm)
    val_loss_m[epoch] = cost(y37_val_m, t37_val)

    # val on wwd
    y37_val_wd = forward(np.transpose(x37_val), wwd)
    val_loss_wd[epoch] = cost_decay(y37_val_wd, t37_val, decay=alpha, w=wwd)

    # val on wwdm
    y37_val_wdm = forward(np.transpose(x37_val), wwdm)
    val_loss_wdm[epoch] = cost_decay(y37_val_wdm, t37_val, decay=alpha, w=wwdm)

# compute results
class_err = classification_error(y37, t37_train)
print("class_err: ", class_err)
print train_loss[n_epochs-1], val_loss[n_epochs-1]

class_err = classification_error(y37_m, t37_train)
print("class_err_m: ", class_err)
print train_loss_m[n_epochs-1], val_loss_m[n_epochs-1]

class_err = classification_error(y37_wd, t37_train)
print("class_err_wd: ", class_err)
print train_loss_wd[n_epochs-1], val_loss_wd[n_epochs-1]

class_err = classification_error(y37_wdm, t37_train)
print("class_err_wdm: ", class_err)
print train_loss_wdm[n_epochs-1], val_loss_wdm[n_epochs-1]


# gradient descent
plt.figure(0)
# plt.title("N=%i iterations" %n_epochs)
plt.subplot(2, 2, 1)
plt.plot(xaxis, train_loss)
plt.plot(xaxis, val_loss)
plt.legend(["training set", "val set"])
plt.title("Gradient Descent (eta=%1.3f)" %eta)
plt.xlabel("N")
plt.ylabel("loss")

# gradient descent with momentum
# plt.figure(1)
plt.subplot(2, 2, 2)
plt.plot(xaxis, train_loss_m)
plt.plot(xaxis, val_loss_m)
plt.legend(["training set", "val set"])
plt.title("Momentum (alpha=%1.3f)" %alpha)
plt.xlabel("N")
plt.ylabel("loss")

# gradient descent with weight decay
# plt.figure(2)
plt.subplot(2, 2, 3)
plt.plot(xaxis, train_loss_wd)
plt.plot(xaxis, val_loss_wd)
plt.legend(["training set", "val set"])
plt.title("Weight decay")
plt.xlabel("N")
plt.ylabel("loss")

# gradient descent with momentum and weight decay
# plt.figure(3)
plt.subplot(2, 2, 4)
plt.plot(xaxis, train_loss_wdm)
plt.plot(xaxis, val_loss_wdm)
plt.legend(["training set", "val set"])
plt.title("Weight decay + momentum (alpha=%1.3f)" %alpha)
plt.xlabel("N")
plt.ylabel("loss")
plt.show()



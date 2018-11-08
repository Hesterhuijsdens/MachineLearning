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
nr_of_batches = 100

# initialize loss arrays
train_loss, train_loss_m, val_loss, val_loss_m, train_loss_wd, val_loss_wd, train_loss_wdm, val_loss_wdm = \
    (np.zeros(epochs) for list in range(8))
xaxis = []

# initialize weights
w, wm, wwd, wwdm = (np.random.randn(1, d) for weights in range(4))

# initialize predictions
y, ym, ywd, ywdm = (np.zeros((1, d)) for i in range(4))

# initialize gradients
dW, dWm, dWwd, dWwdm = (np.random.randn(1) for i in range(4))

# Start time:
start = time.time()

for epoch in range(epochs):
    print "Epoch: ", epoch

    for batch_nr in range(0, N, nr_of_batches):
        # get minibatch:
        batch_x = x[batch_nr:batch_nr + nr_of_batches]
        batch_t = t[batch_nr:batch_nr + nr_of_batches]
        print np.shape(batch_x)
        print np.shape(w)
        print " "
        y[0, batch_nr:batch_nr + nr_of_batches] = forward(np.transpose(batch_x), w)
        ym[0, batch_nr:batch_nr + nr_of_batches] = forward(np.transpose(batch_x), wm)
        ywd[0, batch_nr:batch_nr + nr_of_batches] = forward(np.transpose(batch_x), wwd)
        ywdm[0, batch_nr:batch_nr + nr_of_batches] = forward(np.transpose(batch_x), wwdm)

        # backward propagation:
        gradE = backward(batch_x, y[0, batch_nr:batch_nr + nr_of_batches], batch_t)
        gradE_m = backward(batch_x, ym[0, batch_nr:batch_nr + nr_of_batches], batch_t)
        gradE_wd = gradient_e_decay(ywd[0, batch_nr:batch_nr + nr_of_batches], batch_t, batch_x, decay, wwd)
        gradE_wdm = gradient_e_decay(ywd[0, batch_nr:batch_nr + nr_of_batches], batch_t, batch_x, decay, wwdm)

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
    print "y: ", np.shape(y)
    print "t: ", np.shape(t)
    train_loss[epoch] = cost(y, t)
    train_loss_m[epoch] = cost(ym, t)
    train_loss_wd[epoch] = cost_decay(ywd, t, decay, wwd)
    train_loss_wdm[epoch] = cost_decay(ywdm, t, decay, wwdm)
    xaxis.append(epoch)

    # val on w
    y_test = forward(np.transpose(x_test), w)
    val_loss[epoch] = cost(y_test, t_test)

    # # val on wm
    y_testm = forward(np.transpose(x_test), wm)
    val_loss_m[epoch] = cost(y_testm, t_test)

    # val on wwd
    y_testwd = forward(np.transpose(x_test), wwd)
    val_loss_wd[epoch] = cost_decay(y_testwd, t_test, decay, wwd)

    # val on wwdm
    y_testwdm = forward(np.transpose(x_test), wwdm)
    val_loss_wdm[epoch] = cost_decay(y_testwdm, t_test, decay, wwdm)

# stop time:
end = time.time()
print "time: ", end - start

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




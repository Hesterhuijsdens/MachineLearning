import numpy as np
import matplotlib.pyplot as plt
from PreprocessData import load
from Equations import *
import time
from sklearn.metrics import log_loss

# load train and test data:
x, t = load(version="train")
x = x[:300]
t = t[:300]
x_test, t_test = load(version="test")
x_test = x_test[:100]
t_test = t_test[:100]

# store dimensions of data:
N = np.shape(x)[0]
d = np.shape(x)[1]
w = np.random.rand(1, d)

# set parameters:
decay = 0.1
epochs = 10
losses = []
losses_test = []
xaxis = []

# Start time:
start = time.time()

dW = 0
for epoch in range(epochs):
    print "Epoch: ", epoch

    # forward computation and losses:
    y = forward(np.transpose(x), w)
    losses.append(log_loss(t, y[0]))
    #losses.append(cost(y, t))
    y_test = forward(np.transpose(x_test), w)
    losses_test.append(log_loss(t_test, y_test[0]))
   # losses_test.append(cost(y_test, t_test))
    xaxis.append(epoch)

    # compute gradient and hessian and update the weights:
    grE = gradient_e_decay(y, t, x, decay, w)
    H = hessian(x, y, decay)
    H_inv = np.linalg.pinv(H.astype(float))
    H_grE = np.transpose(np.matmul(H_inv, np.transpose(grE)))

    # use momentum to update weights
    dW = -0.05 * H_grE + 0.1 * dW
    w = w + dW

# compute and plot results:
class_err = classification_error(y, t)
print("class_err: ", class_err)
print("E: ", cost(y, t))


class_err_test = classification_error(forward(np.transpose(x_test), w), t_test)
print("class_err_test: ", class_err_test)
print("E test: ", cost(forward(np.transpose(x_test), w), t_test))

# stop time:
end = time.time()
print("time: ", end - start)

plt.figure(1)
plt.plot(xaxis, losses, label='train')
plt.plot(xaxis, losses_test, label='test')
plt.xlabel('number of epochs')
plt.ylabel('loss')
plt.title('Loss for Newton method with decay = ' + str(decay))
plt.legend()
plt.show()
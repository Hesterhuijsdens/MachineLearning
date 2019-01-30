import numpy as np
import matplotlib.pyplot as plt
from PreprocessData import load37
from Equations import *
import time

# load train and test data:
x, t = load37(version="train")
x_test, t_test = load37(version="test")

# store dimensions of data:
N = np.shape(x)[0]
d = np.shape(x)[1]
w = np.random.randn(1, d)
direction = np.random.randn(d)

# set parameters:
decay = 0.1
epochs = 250
losses = []
xaxis = []

# Start time:
start = time.time()
epoch = 0
step_size = 10
direction = 5

while np.sum(abs(step_size * direction)) > 0.3:
    if epoch % 10 == 0:
        print "Epoch: ", epoch

    # forward computation and losses:
    y = forward(np.transpose(x), w)
    losses.append(cost_decay(y, t, decay, w))
    xaxis.append(epoch)

    gradient_decay = gradient_e_decay(y, t, x, decay, w)
    if epoch > 0:
        # compute the direction:
        steepest_direction = -1.0 * gradient_decay

        # Polak-Ribiere rule:
        beta = polak_ribiere(old_decay, gradient_decay)

        # update the direction:
        direction = steepest_direction + beta * direction
    else:
        direction = -1.0 * gradient_decay
    step_size = golden_section_search(0, 3, w, x, t, direction)

    # update weights:
    old_decay = gradient_decay
    w = w + step_size * direction
    epoch += 1

# compute and plot results:
print " "
print "number of epochs: ", epoch
class_err = classification_error(y, t)
print "class_err: ", class_err
print "E (decay): ", cost_decay(y, t, decay, w)
print "E :", cost(y, t)

class_err_test = classification_error(forward(np.transpose(x_test), w), t_test)
print "class_err_test: ", class_err_test
print "E test (decay): ", cost_decay(forward(np.transpose(x_test), w), t_test, decay, w)
print "E test: ", cost(forward(np.transpose(x_test), w), t_test)

# stop time:
end = time.time()
print "time: ", end - start

# compute test results:
y_test = forward(np.transpose(x_test), w)
loss_test = cost_decay(y_test, t_test, decay, w)

plt.figure(1)
plt.plot(xaxis, losses, label='train', color='royalblue')
plt.plot(xaxis, np.repeat(loss_test, epoch), label='test', color='grey', linestyle='--')
plt.title('Conjugate gradient descent')
plt.xlabel('number of epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


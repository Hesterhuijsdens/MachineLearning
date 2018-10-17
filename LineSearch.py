import numpy as np
import matplotlib.pyplot as plt
from PreprocessData import load
from Equations import *
import time
from scipy.optimize import line_search, minimize


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
# decay = 0.1
epochs = 200
losses = []
losses_test = []
xaxis = []

# Start time:
start = time.time()

for epoch in range(epochs):

    # forward computation and losses:
    y = forward(np.transpose(x), w)
    losses.append(cost(y, t))
    y_test = forward(np.transpose(x_test), w)
    losses_test.append(cost(y_test, t_test))
    xaxis.append(epoch)

    # line search computation:
    direction = -1.0 * backward(x, y, t)
    step_size = golden_section_search(0, 10, w, x, t, direction)

    # update weights:
    w = w + step_size * direction

# compute and plot results:
class_err = classification_error(y, t)
print("class_err: ", class_err)

class_err_test = classification_error(forward(np.transpose(x_test), w), t_test)
print("class_err_test: ", class_err_test)

# stop time:
end = time.time()
print("time: ", end - start)

plt.figure(1)
plt.plot(xaxis, losses, label='train')
plt.plot(xaxis, losses_test, label='test')
plt.xlabel('number of epochs')
plt.ylabel('loss')
plt.legend()
plt.show()
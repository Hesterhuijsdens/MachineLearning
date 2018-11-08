import numpy as np
import matplotlib.pyplot as plt
from PreprocessData import load37
from Equations import *
import time

# load train and test data:
x, t = load37(version="train")
#x = x[:300]
#t = t[:300]
x_test, t_test = load37(version="test")
#x_test = x_test[:100]
#t_test = t_test[:100]

# store dimensions of data:
N = np.shape(x)[0]
d = np.shape(x)[1]
w = np.random.rand(1, d)

# set parameters:
decay = 0.1
epochs = 250
losses = []
losses_test = []
xaxis = []

# Start time:
start = time.time()

for epoch in range(epochs):
    if epoch % 10 == 0:
        print "Epoch: ", epoch

    # forward computation and losses:
    y = forward(np.transpose(x), w)
    losses.append(cost_decay(y, t, decay, w))
    y_test = forward(np.transpose(x_test), w)
    losses_test.append(cost_decay(y_test, t_test, decay, w))
    xaxis.append(epoch)

    # line search computation:
    direction = -1.0 * gradient_e_decay(y, t, x, decay, w)
    step_size = golden_section_search(0, 10, w, x, t, direction)

    # update weights:
    w = w + step_size * direction

# compute and plot results:
class_err = classification_error(y, t)
print("class_err: ", class_err)
print("E: ", cost_decay(y, t, decay, w))

class_err_test = classification_error(forward(np.transpose(x_test), w), t_test)
print("class_err_test: ", class_err_test)
print("E test: ", cost_decay(forward(np.transpose(x_test), w), t_test, decay, w))

# stop time:
end = time.time()
print("time: ", end - start)

plt.figure(1)
plt.plot(xaxis, losses, label='train')
plt.plot(xaxis, losses_test, label='test')
plt.title('Loss for line search')
plt.xlabel('number of epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from PreprocessData import load
from Equations import *

x, t = load(version="train")   # Load the data
x = x[:10]
t = t[:10]
print(t)

x_test, t_test = load(version="test")
x_test = x_test[:10]
t_test = t_test[:10]

N = np.shape(x)[0]
d = np.shape(x)[1]
w = np.random.rand(1, d)

# lambda
decay = 0.1
epochs = 3
losses = []
losses_test = []
xaxis = []

for epoch in range(epochs):
    print("Epoch: ", epoch)
    y = forward(np.transpose(x), w)
    print y
    grE = gradient_e_decay(y, t, x, decay, w)
    H = hessian(x, y, decay, d)
    print np.linalg.det(H)
    H_inv = np.linalg.pinv(H)
    w = w + np.transpose(-1 * np.matmul(H_inv, np.transpose(grE)))
    print("loss: ", cost(y, t))
    losses.append(cost(y, t))
    losses_test.append(cost(forward(np.transpose(x_test), w), t_test))
    xaxis.append(epoch)


# Compute and plot results:
class_err = classification_error(y, t)
print("class_err: ", class_err)

plt.figure(1)
plt.plot(xaxis, losses, label='train')
plt.plot(xaxis, losses_test, label='test')
plt.xlabel('number of epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


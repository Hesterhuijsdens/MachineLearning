import numpy as np
import matplotlib.pyplot as plt
from PreprocessData import load
from Equations import *

x, t = load()   # Load the data
x = x[:10]
t = t[:10]
N = np.shape(x)[0]
d = np.shape(x)[1]
w = np.random.randn(1, d)
decay = 0.1 # lambda

epochs = 10
losses = []
xaxis = []

for epoch in range(epochs):
    y = forward(np.transpose(x), w)        # y will be 1x100
    grE = gradient_e_decay(N, y, t, x, decay, w)  # 1x785
    H = hessian(N, x, y, decay, d)
    Hinv = np.linalg.inv(H)        # should be changed to real inverse
    w = w + np.transpose(-1 * np.matmul(Hinv, np.transpose(grE)))
    print("loss: ", cost(y, t, N))
    losses.append(cost(y, t, N))
    xaxis.append(epoch)

# Compute results:
class_err = classification_error(y, t)
print("class_err: ", class_err)

plt.figure(1)
plt.plot(xaxis, losses)
plt.show()



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

# lambda
decay = 0.1

epochs = 10
losses = []
xaxis = []

for epoch in range(epochs):
    y = forward(np.transpose(x), w)
    grE = gradient_e_decay(y, t, x, decay, w)  # 1x785
    print("grE")
    print np.shape(grE)
    print grE

    H = hessian(x, y, decay, d)
    print np.max(H)
    print np.shape(H)
    print np.linalg.det(H)
    Hinv = np.linalg.pinv(H)
    w = w + np.transpose(-1 * np.matmul(Hinv, np.transpose(grE)))
    print("loss: ", cost(y, t))
    losses.append(cost(y, t))
    xaxis.append(epoch)

# Compute results:
class_err = classification_error(y, t)
print("class_err: ", class_err)

plt.figure(1)
plt.plot(xaxis, losses)
plt.show()



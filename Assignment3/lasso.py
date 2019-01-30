import numpy as np
from load import loadLasso, normalize
import matplotlib.pyplot as plt
from equations_regression import *

np.random.seed(40)

# load lasso data and normalize:
X, y = loadLasso(version="train")   # (50, 100) & (50,)
Xv, yv = loadLasso(version="test")
X = normalize(X)
Xv = normalize(Xv)

# add bias term (column of ones):
x = np.ones((np.shape(X)[0], np.shape(X)[1] + 1))
x[:, 1:np.shape(X)[1] + 1] = X
xv = np.ones((np.shape(Xv)[0], np.shape(Xv)[1] + 1))
xv[:, 1:np.shape(Xv)[1] + 1] = Xv

# store data dimensions:
p = np.shape(x)[0] # samples -> 50
n = np.shape(x)[1] # dimensions -> 101
p_test = np.shape(xv)[0]

# set regularization term:
decay = 0.01#0.000000000000000000001 #0.018 #0.01

# initialization for LASSO:
cost = []
w = np.random.normal(loc=0, scale=0.1, size=n)
w_plot = np.zeros((n, 1000))
cost_test = []

# coordinate descent:
epoch = 0
converged = False

while not converged or epoch < 10:
    # save w for plotting:
    print epoch
    w_plot[:, epoch] = w

    converged = True

    for i in range(1, n): # for all dimensions (100)
        # compute gradient of one single feature i and update:
        ytilde = y - np.matmul(w, np.transpose(x)) + w[i] * x[:, i]
        btilde = (1.0/p) * np.dot(ytilde, x[:, i])
        w_new = soft_threshold(btilde, decay) #/(x[:, i]**2).sum()

        if np.abs(w_new - w[i]) >= 0.000001:
            converged = False
        w[i] = 0.8 * w_new + 0.2 * w[i]

    cost.append(cost_function(p, w, x, y, decay))
    cost_test.append(cost_function(p_test, w, xv, yv, decay))
    epoch += 1

print "epochs: ", epoch

# plot w as function of epoch:
plt.figure(1)
#plt.plot(range(epoch), np.transpose(np.transpose(w_plot)[1:101]))
plt.plot(range(epoch), np.transpose(w_plot[1:101, 0:epoch]))
plt.xlabel('iteration')
plt.ylabel('coefficients w')
plt.title('LASSO using coordinate descent')
plt.show()


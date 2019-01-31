import numpy as np
from load import loadLasso, normalize
import matplotlib.pyplot as plt
from equations_regression import *

np.random.seed(40)

# load lasso data and normalize:
X, y = loadLasso(version="train")
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

# set regularization term:
alpha = np.logspace(-2, 0.5, 100)

# initialize for cross fold validation:
cost_crossfold = np.zeros(len(alpha))
cost_crossfold_test = np.zeros(len(alpha))

counter = 0
# cross validation:
for decay in alpha:
    if counter == 0:
        w = np.random.normal(loc=0, scale=0.1, size=n)

    # coordinate descent:
    epoch = 0
    converged = False
    while not converged or epoch < 10:
        converged = True
        for i in range(1, n): # for all dimensions (100)
            # compute gradient of one single feature i and update:
            ytilde = y - np.matmul(w, np.transpose(x)) + w[i] * x[:, i]
            btilde = np.dot(ytilde, x[:, i])
            w_new = soft_threshold(btilde, decay)
            if np.abs(w_new - w[i]) >= 0.000001:
                converged = False
            w[i] = 0.8 * w_new + 0.2 * w[i]
        epoch += 1

    cost_crossfold[counter] = cost_function(p, w, x, y)
    cost_crossfold_test[counter] = cost_function(p, w, xv, yv)
    counter += 1

# plot costs (after convergence): as function of decay value:
plt.figure(2)
#plt.plot(alpha, cost_crossfold, label='train', color='orange')
plt.plot(alpha, cost_crossfold_test, label='validation', color='royalblue')
plt.xlabel(r'$\lambda$')
plt.ylabel('error')
plt.title('cross validation with LASSO')
plt.legend()

plt.show()


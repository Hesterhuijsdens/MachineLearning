import numpy as np
from load import loadLasso, normalize2
import matplotlib.pyplot as plt
from equations_regression import *
from sklearn.linear_model import Ridge

# load lasso data and normalize:
X, y = loadLasso(version="train")   # (50, 100) & (50,)
Xv, yv = loadLasso(version="test")
X = normalize2(X)
Xv = normalize2(Xv)

print np.mean(X)
print np.mean(Xv)

# add bias term (column of ones):
x = np.ones((np.shape(X)[0], np.shape(X)[1] + 1))
x[:, 1:np.shape(X)[1] + 1] = X
xv = np.ones((np.shape(Xv)[0], np.shape(Xv)[1] + 1))
xv[:, 1:np.shape(Xv)[1] + 1] = Xv

# store data dimensions:
p = np.shape(x)[0] # samples -> 50
n = np.shape(x)[1] # dimensions -> 101

# set regularization term:
decay = 1.0
eta = 1.0

# initialization for LASSO:
max_iteration = 40
cost = np.zeros(max_iteration)
w = np.random.rand(n)
w_plot = np.zeros((n, max_iteration))
cost_test = np.zeros(max_iteration)

# coordinate descent:
for epoch in range(max_iteration):
    # save w for plotting:
    w_plot[:, epoch] = w

    for i in range(1, n): # for all dimensions (100)
        # compute gradient of one single feature i and update:
        ytilde = y - np.matmul(w, np.transpose(x))
        btilde = np.dot(ytilde, x[:, i])
        w[i] += eta * soft_threshold(btilde, decay)/(x[:, i]**2).sum()

    cost[epoch] = cost_function(p, w, x, y, decay)
    cost_test[epoch] = cost_function(p, w, xv, yv, decay)

print "cost lasso (training): ", cost_function(p, w, x, y, decay)

# print "cost ridge: ", cost_function(p, w_ridge, x, y, decay)
# ridge = Ridge(alpha=1.0, fit_intercept=False).fit(x, y)
# print "cost 2 ridge: ", cost_function(p, ridge.coef_, x, y, decay)

# plot the costs of lasso as function of epoch:
plt.figure(1)
plt.plot(range(max_iteration), cost, label="train")
plt.plot(range(max_iteration), cost_test, label="test")
plt.legend()

# plot w as function of epoch:
plt.figure(2)
plt.plot(range(max_iteration), np.transpose(w_plot))
plt.show()


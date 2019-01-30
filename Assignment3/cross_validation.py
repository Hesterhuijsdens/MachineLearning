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

# add bias term (column of ones):
x = np.ones((np.shape(X)[0], np.shape(X)[1] + 1))
x[:, 1:np.shape(X)[1] + 1] = X
xv = np.ones((np.shape(Xv)[0], np.shape(Xv)[1] + 1))
xv[:, 1:np.shape(Xv)[1] + 1] = Xv

# store data dimensions:
p = np.shape(x)[0] # samples -> 50
n = np.shape(x)[1] # dimensions -> 101

# set regularization term:
#alpha = np.linspace(0.01, 10, 100)
alpha = np.logspace(-2, 1.5, 100)

# initialization for LASSO:
max_iteration = 60
cost = np.zeros(max_iteration)
w_plot = np.zeros((n, max_iteration))
cost_test = np.zeros(max_iteration)

# initialize for cross fold validation:
cost_crossfold = np.zeros(len(alpha))
cost_crossfold_test = np.zeros(len(alpha))

counter = 0
# cross validation:
for decay in alpha:
    if counter == 0:
        w = np.random.normal(loc=0, scale=0.1, size=n) #np.random.rand(n)
        max_iteration = 100

    # coordinate descent:
    for epoch in range(max_iteration):
        # save w for plotting:
        #w_plot[:, epoch] = w

        for i in range(1, n): # for all dimensions (100)
            # compute gradient of one single feature i and update:
            ytilde = y - np.matmul(w, np.transpose(x)) + w[i] * x[:, i]
            btilde = np.dot(ytilde, x[:, i])
            w[i] = soft_threshold(btilde, decay)/(x[:, i]**2).sum()

     #   cost[epoch] = cost_function(p, w, x, y, decay)
     #   cost_test[epoch] = cost_function(p, w, xv, yv, decay)

    cost_crossfold[counter] = cost_function(p, w, x, y, decay)
    cost_crossfold_test[counter] = cost_function(p, w, xv, yv, decay)
    counter += 1

#print "cost lasso (training): ", cost_function(p, w, x, y, decay)

# print "cost ridge: ", cost_function(p, w_ridge, x, y, decay)
# ridge = Ridge(alpha=1.0, fit_intercept=False).fit(x, y)
# print "cost 2 ridge: ", cost_function(p, ridge.coef_, x, y, decay)

# plot the costs of lasso as function of epoch:
# plt.figure(1)
# plt.plot(range(max_iteration), cost, label="train")
# plt.plot(range(max_iteration), cost_test, label="test")
# plt.xlabel('epoch')
# plt.ylabel('cost')
# plt.legend()
#
# # plot w as function of epoch:
# plt.figure(2)
# plt.plot(range(max_iteration), np.transpose(w_plot))
# plt.xlabel('epoch')
# plt.ylabel('value of w')

# plot costs (after max_iteration epochs) as function of decay value:
plt.figure(3)
plt.plot(alpha, cost_crossfold, label='train')
plt.plot(alpha, cost_crossfold_test, label='test')
plt.xlabel(r'$\lambda$')
plt.ylabel('cost')
plt.legend()

plt.show()

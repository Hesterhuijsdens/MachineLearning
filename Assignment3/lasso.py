import numpy as np
from load import loadLasso
import matplotlib.pyplot as plt
import time
from equations_regression import *

# load lasso data:
x, y = loadLasso(version="train")   # (50, 100) & (50,)
xv, yv = loadLasso(version="test")

# store data dimensions:
p = np.shape(x)[0] # samples -> 50
n = np.shape(x)[1] # dimensions -> 100

# set regularization term:
decay = 0.1
eta = 1.0

# OLS:
w = np.matmul(np.linalg.inv(chi(x)), b(x, y))
print ordinary_least_squares(y, x, w)

# ridge regression:
w = np.matmul(np.linalg.inv(chi(x) + decay * np.eye(n)), b(x, y))
print np.shape(w)
print ols_ridge(y, x, w, decay)
print ordinary_least_squares(y, x, w)

# lasso: zie slide 68
#print beta_estimate(b(x, y), chi(x), w)
for epoch in range(20):
    print epoch
    index = np.shape(w)[0] % (epoch + 1)

    w[index] += -eta * gradient_error(p, y, w, x, epoch)
    print ols_lasso(y, x, w, decay)










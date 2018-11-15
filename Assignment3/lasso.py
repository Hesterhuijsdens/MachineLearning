import numpy as np
from load import loadLasso
import matplotlib.pyplot as plt
import time
from equations_regression import *

# load lasso data:
x, y = loadLasso(version="train")   # (50, 100) & (50,)
xv, yv = loadLasso(version="test")

# store data dimensions:
p = np.shape(x)[0] # samples
n = np.shape(x)[1] # dimensions

# set regularization term:
gamma = 0.1

# OLS:
w = np.matmul(np.linalg.inv(chi(x)), b(x, y))
print ordinary_least_squares(y, x, w)

# ridge regression:
w = np.matmul(np.linalg.inv(chi(x) + 0.1 * np.eye(n)), b(x, y))
print np.shape(w)
print ols_ridge(y, x, w, 0.1)
print ordinary_least_squares(y, x, w)

# lasso:
w =





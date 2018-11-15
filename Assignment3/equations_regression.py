import numpy as np


# function to compute OLS:
def ordinary_least_squares(y, x, w):
    return np.sum(pow(y - np.dot(x, w), 2))


# function to compute OLS with regularization term:
def ols_ridge(y, x, w, gamma):
    ols = ordinary_least_squares(y, x, w)
    return ols + gamma * np.dot(w, w)


# function to compute OLS with regularization and linear constraint:
def ols_lasso(y, x, w, gamma):
    ols = ordinary_least_squares(y, x, w)
    return ols + gamma * np.linalg.norm(w, ord=1)


# Xij = 1/p sum(xi * xj) -> (100, 100)
def chi(x):
    p = np.shape(x)[0]
    return (1.0/p) * np.matmul(np.transpose(x), x)


# b_i = 1/p sum(xi * y) -> (100,)
def b(x, y):
    p = np.shape(x)[0]
    return (1.0/p) * np.matmul(np.transpose(x), y)



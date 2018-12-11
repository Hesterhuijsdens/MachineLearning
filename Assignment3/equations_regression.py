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


def beta(beta, gamma):
    return 0

#
def beta_estimate(b, chi, beta):
    array_len = np.shape(chi)[0]
    beta_pred = np.zeros(array_len)
    for j in range(array_len):
        sum = 0
        for i in range(array_len):
            if (i != j) & (beta[i] != 0.0):
                sum += chi[i, j] * beta[i]
        beta_pred[j] = beta[j] - sum
    return beta_pred


def gradient_error(N, y, beta, x):
    test = np.matmul(y - np.matmul(x, beta), x)
    return (-1.0 / N) * test


    # y = (50,), beta = (100,), x = (50, 100)
    # y - xb = (50, )



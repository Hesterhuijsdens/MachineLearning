import numpy as np


# function to compute OLS:
def ordinary_least_squares(y, x, w):
    return np.sum(np.power(y - np.matmul(x, w), 2))


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


# #
# def beta_estimate(b, chi, w):
#     array_len = np.shape(chi)[0]
#     beta_pred = np.zeros(array_len)
#     for j in range(array_len):
#         sum = 0
#         for i in range(array_len):
#             if (i != j) & (w[i] != 0.0):
#                 sum += chi[i, j] * w[i]
#         beta_pred[j] = w[j] - sum
#     return beta_pred


# def gradient_error_ridge(N, y, w, x, decay):
#     b_i = (1.0 / N) * np.matmul(np.transpose(x), y)  # slide 58
#     Chi = chi(x)  # (100,100)
#     lagrange = decay * (w / np.linalg.norm(w, ord=2))
#     return -1.0 * b_i + np.matmul(Chi, w) + lagrange

#
# def gradient_error_lasso(N, y, w, x, i, decay):
#     b_i = (1.0/N) * np.dot(x[:, i], y) # slide 58
#     Chi = chi(x) # (100,100)
#     #lagrange = decay * np.sign(w[i])
#     return -1.0 * b_i + np.dot(Chi[:, i], w)# + lagrange


# Cost function with L1-norm:
def cost_function(N, w, x, y, decay): # slide 58: f(w)
    cost_normal =  np.sum(np.power(y - np.matmul(w, np.transpose(x)), 2))
    #lagrange = decay * np.sum(np.abs(w))
    return (1.0/(2.0*N)) * cost_normal #+ lagrange

    # y = (50,), w = (100,), x = (50, 100)
    # y - xb = (50, )


# function S(..) used for LASSO updates:
def soft_threshold(b, decay):
    if decay >= abs(b):
        return 0
    else:
        return b - decay * np.sign(b)

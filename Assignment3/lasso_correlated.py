import numpy as np
import matplotlib.pyplot as plt
from load import normalize
from equations_regression import *

np.random.seed(40)


# function to generate (correlated) data:
def correlated_inputs():
    n = 3           # dimensions
    p = 1000        # samples
    #w = [2.0, 3.0, 0.0]
    w = [-2.0, 3.0, 0.0]
    x = np.zeros((p, n))
    x[:, 0:2] = np.random.randn(p, 2)
    x[:, 2] = (2.0/3.0) * x[:, 0] + (2.0/3.0) * x[:, 1] + (1.0/3.0) * np.random.randn(p)
    y = np.matmul(w, np.transpose(x)) + np.random.randn(p)
    return x, y


# load data:
X, y = correlated_inputs()
x = normalize(X)

# store data dimensions:
p = np.shape(x)[0] # samples -> 1000
n = np.shape(x)[1] # dimensions -> 3

# set regularization term:
alpha = np.logspace(-5, 2, 100)

# initialize for cross fold validation:
weights_crossfold = np.zeros((len(alpha), n))

counter = 0
# cross validation:
for decay in alpha:
    if counter == 0:
        w = np.random.normal(loc=0, scale=0.1, size=n)

    # coordinate descent:
    epoch = 0
    converged = False
    while not converged or epoch < 10:
        print epoch
        converged = True
        for i in range(1, n): # for all dimensions (100)
            # compute gradient of one single feature i and update:
            ytilde = y - np.matmul(w, np.transpose(x)) + w[i] * x[:, i]
            btilde = np.dot(ytilde, x[:, i])
            w_new = soft_threshold(btilde, decay)
            if np.abs(w_new - w[i]) >= 0.01:
                converged = False
            w[i] = w_new
        epoch += 1

    weights_crossfold[counter] = w
    counter += 1


plt.figure(4)
plt.plot(alpha, weights_crossfold[:, 0], label=r'$w_1$')
plt.plot(alpha, weights_crossfold[:, 1], label=r'$w_2$')
plt.plot(alpha, weights_crossfold[:, 2], label=r'$w_3$')
plt.xlabel(r'$\lambda$')
plt.ylabel('coefficients')
plt.title('w = [-2, 3, 0]')
plt.legend()
plt.show()
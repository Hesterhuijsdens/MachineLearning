import numpy as np
import matplotlib.pyplot as plt


# function to generate (correlated) data:
def correlated_inputs():
    n = 3           # dimensions
    p = 1000        # samples
    w = [2.0, 3.0, 0.0]
    #w = [-2.0, 3.0, 0.0]
    sigma = 1.0
    x = np.zeros((p, n))
    x[:, 0:2] = np.random.randn(p, 2)
    x[:, 2] = (2.0/3.0) * x[:, 0] + (2.0/3.0) * x[:, 1] + (1.0/3.0) * np.random.randn(p)
    y = np.matmul(w, np.transpose(x)) + np.random.randn(p)
    return x, y


x, y = correlated_inputs()
print np.shape(x)
print np.shape(y)
import numpy as np
from load import loadLasso
import matplotlib.pyplot as plt
import time

# load lasso data:
x, y = loadLasso(version="train")   # (50, 100) & (50,)
xv, yv = loadLasso(version="test")

# store data dimensions:
p = np.shape(x)[0] # samples
n = np.shape(x)[1] # dimensions
#w = np.random.randn(n + 1)


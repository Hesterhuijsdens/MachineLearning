import numpy as np
import matplotlib.pyplot as plt
from PreprocessData import load37
from Equations import *
import time

# load train and test data:
x, t = load37(version="train")
x = x[:300]
t = t[:300]
x_test, t_test = load37(version="test")
x_test = x_test[:100]
t_test = t_test[:100]

# store dimensions of data:
N = np.shape(x)[0]
d = np.shape(x)[1]
w = np.random.randn(1, d)

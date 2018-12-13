import numpy as np
from PreprocessData import load37
from binary_data import *
from BM_equations import *

# avoid overflow warnings
np.seterr(all="ignore")


# generate 60,000 random binary patterns
# patterns = get_random_pattern(N=10, lb=3)
# plt.figure()
# for k in range(0, 6):
#     plt.subplot(3, 2, k+1)
#     plt.imshow(patterns[k])
# plt.show()


patterns = np.loadtxt('data.txt')
train_BM(patterns)

# log likelihood
# L = 1.0/N * np.sum(np.log(p))

# plot change in weights vs iteration

# mean field theory and linear regression correction

# load in MNIST data

# build classifier (2.5.1)










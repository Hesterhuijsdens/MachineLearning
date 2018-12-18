import numpy as np
import matplotlib.pyplot as plt
from PreprocessData import load3
np.random.seed(0)


# generate random binary patterns
def get_binary_mnist(lb):
    x_training = load3(version="train")
    x = x_training[:lb]
    print np.shape(x)
    thr = 0.05
    x_bin = []
    for i in range(lb):

        x_binary = (2 * (x[i] > thr)) - 1

        # add noise
        x_binary = x_binary * (np.random.binomial(1, 0.9, size=np.shape(x_binary))*2-1)

        print x_binary

        x_bin.append(x_binary.tolist())
    return np.transpose(x_bin)
    # return np.transpose(x)


# generate matrix with random patterns
def get_random_pattern(N, lb):
    return np.random.choice((-1., 1.), size=(N, lb))

# plt.figure()
# for k in range(0, 6):
#     plt.subplot(3, 2, k+1)
#     plt.imshow(random_patterns[k])
# plt.show()


# generate random w
def get_w(N):
    w = np.random.uniform(-1., 1., size=(N, N))
    np.fill_diagonal(w, 0)
    return w


# generate random b
def get_b(N):
    return np.random.choice((-1., 1.), size=N)

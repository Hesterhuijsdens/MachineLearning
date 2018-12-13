import numpy as np
import matplotlib.pyplot as plt
from PreprocessData import load37


# generate random binary patterns
def get_binary_mnist(lb):
    x_training, t_training = load37(version="train")
    x = x_training[:lb]
    t = t_training[:lb]
    thr = 0.1
    x_bin = []
    for i in range(lb):
        x_binary = (2 * (x[i] > thr)) - 1
        x_bin.append(x_binary)
    return x_bin


# binary x: -1 or +1
# thresholds = np.linspace(0, 0.1, 5)
# plt.figure()
# plt.subplot(3, 2, 1)
# plt.imshow(np.reshape(x[0, 1:], [28, 28]))
# for i in range(0, 5):
#     x_binary = (2 * (x > thresholds[i])) - 1
#     plt.subplot(3, 2, i+2)
#     plt.imshow(np.reshape(x_binary[0, 1:], [28, 28]))
# plt.show()


# generate matrix with random patterns
def get_random_pattern(N, lb):
    return np.random.choice((-1., 1.), size=(N, lb))


# data = get_random_pattern(5, 50)
# np.savetxt('data.txt', data, fmt="%5.1f")


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


# weight = get_w(10)
# bias = get_b(5)
# np.savetxt('w.txt', weight, fmt="%5.1f")
# np.savetxt('b.txt', bias, fmt="%5.1f")

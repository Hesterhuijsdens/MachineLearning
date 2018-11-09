from mnist import MNIST
import numpy as np
import os
import matplotlib.pyplot as plt


# function to normalize the MNIST data:
def normalize(x, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(x, order, axis))
    l2[l2 == 0] = 1
    return x / np.expand_dims(l2, axis)


# Load mnist data and select 3/7:
def load37(version="train"):
    if version == "train":
        os.chdir("..")
    data = MNIST(os.path.abspath(os.curdir) + "\data")
    if version == "train":
        x, y = data.load_training()
        nr = 12396
    else:
        x, y = data.load_testing()
        nr = 2038

    x3 = [x[i] for i in range(0, len(y)) if y[i] == 3]
    x7 = [x[i] for i in range(0, len(y)) if y[i] == 7]
    y3 = [y[i] for i in range(0, len(y)) if y[i] == 3]
    y7 = [y[i] for i in range(0, len(y)) if y[i] == 7]

    #data3 = [normalize(x3), y3]
    #data7 = [normalize(x7), y7]
    data3 = [list(normalize(x3)), y3]
    data7 = [list(normalize(x7)), y7]

    # for i in range(0, 10):
    #     plt.subplot(5, 2, i+1)
    #     image = np.reshape(data3[0][i], [28, 28])
    #     plt.imshow(image)
    # plt.show()

    data = np.concatenate((data3, data7), axis=1)
    np.take(data, np.random.permutation(np.shape(data)[1]), axis=1, out=data)
    dataX = np.zeros((nr, 784))
    for i in range(nr):
        dataX[i, :] = data[0][i]
    dataX = np.column_stack([np.ones((nr, 1)), dataX])

    t = np.reshape(data[1], (nr, 1))
    t = [1 if label == 3 else 0 for label in t]  # t = 1 if the label is 3, t = 0 if the label is 7

    # Plot data
    # plt.figure()
    # for i in range(0, 10):
    #     plt.subplot(5, 2, i + 1)
    #     if i % 2 == 0:
    #         image = np.reshape(x3[i], [28, 28])
    #         plt.imshow(image)
    #     else:
    #         image = np.reshape(x7[i], [28, 28])
    #         plt.imshow(image)
    # plt.suptitle('Examples of the images 3 and 7 from the MNIST dataset')
    # plt.show()

    return dataX, t

    # return dataX, t


# Load all mnist data:
def load(version="train"):
    if version == "train":
        os.chdir("..")
    data = MNIST(os.path.abspath(os.curdir) + "\data")

    if version == "train":
        x, y = data.load_training()
        nr = 60000
    else:
        x, y = data.load_testing()
        nr = 10000

    data = [normalize(x), y]
#    np.take(data, np.random.permutation(np.shape(data)[1]), axis=1, out=data)

    dataX = np.zeros((nr, 784))
    for i in range(nr):
        dataX[i, :] = data[0][i]
    dataX = np.column_stack([np.ones((nr, 1)), dataX])
    t = np.reshape(data[1], (nr, 1))
    return dataX, t

# Plot data
# for i in range(0, 10):
#     plt.subplot(5, 2, i+1)
#     image = np.reshape(x7[i], [28, 28])
#     plt.imshow(image)
# plt.show()



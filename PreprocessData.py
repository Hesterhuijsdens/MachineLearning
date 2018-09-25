from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt


def normalize(x): # Rescale to [0,1] range
    min = np.min(x, 0)
    max = np.max(x, 0)
    for i in range(0, len(min)):
        if (max[i]-min[i]) == 0:
            max[i] = 1
            min[i] = 0
    return [(x[i]-min)/(max-min) for i in range(0, np.shape(x)[0])]


# Load mnist data and select 3/7:
def load(version="train"):
    data = MNIST('data')
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

    data3 = [normalize(x3), y3]
    data7 = [normalize(x7), y7]
    data = np.concatenate((data3, data7), axis=1)
    np.take(data, np.random.permutation(np.shape(data)[1]), axis=1, out=data)
    dataX = np.zeros((nr, 784))
    for i in range(nr):
        dataX[i, :] = data[0][i]
    dataX = np.column_stack([np.ones((nr, 1)), dataX])

    t = np.reshape(data[1], (nr, 1))
    t = [1 if label == 3 else 0 for label in t]  # t = 1 if the label is 3, t = 0 if the label is 7
    return dataX, t


# Plot data
# for i in range(0, 10):
#     plt.subplot(5, 2, i+1)
#     image = np.reshape(x7[i], [28, 28])
#     plt.imshow(image)
# plt.show()


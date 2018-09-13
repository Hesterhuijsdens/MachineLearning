from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
from PreprocessData import load


x, y = load()
print(np.shape(x))
print(np.shape(y))

# Plot data
for i in range(0, 10):
    plt.subplot(5, 2, i+1)
    image = np.reshape(x[i], [28, 28])
    plt.imshow(image)
plt.show()





def sigmoid(x):
    return (1+np.exp(-x))**(-1)


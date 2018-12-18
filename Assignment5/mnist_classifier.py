import numpy as np
from binary_data import *
from equations_BM import *

x = get_binary_mnist(10)
plt.figure()
for i in range(0, 6):
    plt.subplot(3, 2, i+1)
    plt.imshow(np.reshape(np.transpose(x)[i], [28, 28]))
plt.show()

n = 200
# w, b, weightlist = boltzmann_train(x, eta=0.001, n_epochs=n)






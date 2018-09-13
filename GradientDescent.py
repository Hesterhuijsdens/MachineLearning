from mnist import MNIST
import numpy as np

# Load mnist data and select 3/7
data = MNIST('data')
x, y = data.load_training()
x3 = [x[i] for i in range(0,len(y)) if y[i]==3]
x7 = [x[i] for i in range(0,len(y)) if y[i]==7]
# xtest, ytest = data.load_testing()

print(np.shape(x7))
print(np.shape(x3))

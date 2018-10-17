import numpy as np
import matplotlib.pyplot as plt
from PreprocessData import load
from Equations import *
import time

x, t = load(version="train")   # Load the data
x = x[:500]
t = t[:500]
x_test, t_test = load(version="test")
x_test = x_test[:500]
t_test = t_test[:500]

N = np.shape(x)[0]
d = np.shape(x)[1]
w = np.random.rand(1, d)

# lambda
# decay = 0.1
epochs = 10
losses = []
losses_test = []
xaxis = []

# Start time:
start = time.time()

for epoch in range(epochs):
    xaxis.append(epoch)



# Compute and plot results:
class_err = classification_error(y, t)
print("class_err: ", class_err)

class_err_test = classification_error(forward(np.transpose(x_test), w), t_test)
print("class_err_test: ", class_err_test)

# Stop time:
end = time.time()
print("time: ", end - start)

plt.figure(1)
plt.plot(xaxis, losses, label='train')
plt.plot(xaxis, losses_test, label='test')
plt.xlabel('number of epochs')
plt.ylabel('loss')
plt.legend()
plt.show()






import matplotlib.pyplot as plt
from PreprocessData import load37
from Equations import *


# avoid overflow warnings
np.seterr(all="ignore")

# load train (N=12396L) and test (N=2038) data
x_training, t_training = load37(version="train")
x_test, t_test = load37(version="test")

# split 7:1 ratio
lb = np.round(np.shape(x_training)[0] * 7 / 8)
ub = np.shape(x_training)[0]
x = x_training[:lb]
t = t_training[:lb]
x_val = x_training[lb+1:ub]
t_val = t_training[lb+1:ub]

# parameters
eta = 0.03
nr_of_batches = 100

# regular gradient descent, stats: train - val - test
tloss, ytrain, vloss, yval, weight_vector, end, start, n_epochs = training_stochastic(x, t, x_val, t_val, lb, nr_of_batches, 0, 0, eta)
ytest = forward(np.transpose(x_test), weight_vector)
tcost = cost(ytest, t_test)

print "Loss: ", tloss[n_epochs-1], vloss[n_epochs-1], tcost
print "Error: ", classification_error(ytrain, t), classification_error(yval, t_val), classification_error(ytest, t_test)
print "Acc: ", testing(x, weight_vector, t), "%", testing(x_val, weight_vector, t_val), "%", testing(x_test, weight_vector, t_test), "%"
print "Time: ", end - start
print "itr", n_epochs

# gradient descent
plt.figure()
xaxis = range(0, n_epochs)
plt.plot(xaxis, tloss, c='royalblue')
plt.plot(xaxis, vloss, c='darkorange')
plt.plot(xaxis, [tcost] * n_epochs, c='grey', linestyle='--')
plt.legend(["train", "validation", "test"])
plt.title("Stochastic gradient descent (eta=%1.2f)" % eta)
plt.xlabel("N")
plt.ylabel("loss")
plt.show()


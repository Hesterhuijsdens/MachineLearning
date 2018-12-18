import matplotlib.pyplot as plt
from PreprocessData import load37
from Equations import *


# avoid overflow warnings
np.seterr(all="ignore")

# load train (N=12396L) and test (N=2038) data
x_training, t_training = load37(version="train")
x_test, t_test = load37(version="test")

# 80 train : 20 validation
# lb = 9917
# ub = np.shape(x_training)[0] - 1
lb = 8000
ub = 10000
x = x_training[:lb]
t = t_training[:lb]
x_val = x_training[lb+1:ub]
t_val = t_training[lb+1:ub]

# parameters
n_epochs = 300
eta = 1
alpha = 0.8
N = np.shape(x)
xaxis = range(0, n_epochs)
decay = 0.1

# initialize weights and gradients
w, wm, wwd, wwdm = (np.random.randn(1, np.shape(x)[1]) for weights in range(4))
dW, dWm, dWwd, dWwdm = (np.random.randn(1) for i in range(4))

# regular gradient descent
tloss, ytrain, vloss, yval, weight_vector, end, start = training(n_epochs, x=x, t=t, x_val=x_val, t_val=t_val,
                                                                 w=w, dw=dW, momentum=0, weightdecay=0)
class_err = classification_error(ytrain, t)
print "GD2, class error: ", class_err, "val class error", classification_error(yval, t_val)
print "train loss: ", tloss[n_epochs-1], "val loss: ", vloss[n_epochs-1]
ytest = forward(np.transpose(x_test), weight_vector)
tcost = cost(ytest, t_test)
print "test loss: ", tcost

print "train accuracy: ", testing(x, weight_vector, t), "%"
print "test accuracy: ", testing(x_test, weight_vector, t_test), "%"
print "time: ", end - start
print " "

# momentum
tloss_m, ytrain_m, vloss_m, yval_m, weight_vector_m, end_m, start_m = training(n_epochs, x=x, t=t, x_val=x_val,
                                                                               t_val=t_val, w=wm, dw=dWm, alpha=alpha,
                                                                               momentum=1, weightdecay=0)
class_err_m = classification_error(ytrain_m, t)
print "GD+M2, class error: ", class_err_m, "val class error", classification_error(yval_m, t_val)
print "train loss: ", tloss_m[n_epochs-1], "val loss: ", vloss_m[n_epochs-1]
print "test accuracy: ", testing(x_test, weight_vector_m, t_test), "%"
print "time: ", end_m - start_m
print " "

# weight decay
tloss_wd, ytrain_wd, vloss_wd, yval_wd, weight_vector_wd, end_wd, start_wd = training(n_epochs, x=x, t=t, x_val=x_val,
                                                                                      t_val=t_val, w=wwd, dw=dWwd,
                                                                                      decay=decay, momentum=0,
                                                                                      weightdecay=1)
class_err_wd = classification_error(ytrain_wd, t)
print "GD+WD2, class error: ", class_err_wd, "val class error", classification_error(yval_wd, t_val)
print "train loss: ", tloss_wd[n_epochs-1], "val loss: ", vloss_wd[n_epochs-1]
print "test accuracy: ", testing(x_test, weight_vector_wd, t_test), "%"
print "time: ", end_wd - start_wd
print " "

# weight decay + momentum
tloss_wdm, ytrain_wdm, vloss_wdm, yval_wdm, weight_vector_wdm, end_wdm, start_wdm = training(n_epochs, x=x, t=t,
                                                                                             x_val=x_val, t_val=t_val,
                                                                                             w=wwdm, dw=dWwdm,
                                                                                             alpha=alpha, decay=decay,
                                                                                             momentum=1, weightdecay=1)
class_err_wdm = classification_error(ytrain_wdm, t)
print "GD+WD+M2, class error: ", class_err_wdm, "val class error", classification_error(yval_wdm, t_val)
print "train loss: ", tloss_wdm[n_epochs-1], "val loss: ", vloss_wdm[n_epochs-1]
print "test accuracy: ", testing(x_test, weight_vector_wdm, t_test), "%"
print "time: ", end_wdm - start_wdm
print " "

# gradient descent
plt.figure()
plt.subplot(2, 2, 1)
plt.plot(xaxis, tloss)
plt.plot(xaxis, vloss)
plt.legend(["train", "test"])
plt.title("Gradient descent (eta=%1.2f)" %eta)
plt.xlabel("N")
plt.ylabel("loss")

# gradient descent with momentum
plt.subplot(2, 2, 2)
plt.plot(xaxis, tloss_m)
plt.plot(xaxis, vloss_m)
plt.legend(["train", "test"])
plt.title("Momentum (alpha=%1.2f)" %alpha)
plt.xlabel("N")
plt.ylabel("loss")

# gradient descent with weight decay
plt.subplot(2, 2, 3)
plt.plot(xaxis, tloss_wd)
plt.plot(xaxis, vloss_wd)
plt.legend(["train", "test"])
plt.title("Weight decay(decay=%1.2f)" %decay)
plt.xlabel("N")
plt.ylabel("loss")

# gradient descent with momentum and weight decay
plt.subplot(2, 2, 4)
plt.plot(xaxis, tloss_wdm)
plt.plot(xaxis, vloss_wdm)
plt.legend(["train", "test"])
plt.title("Weight decay (decay=%1.3f) + momentum (alpha=%1.2f)".format(decay, alpha))
plt.xlabel("N")
plt.ylabel("loss")
plt.suptitle("Gradient Descent over %i epochs" %n_epochs)
plt.show()




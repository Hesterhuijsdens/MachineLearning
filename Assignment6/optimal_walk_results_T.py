import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)

# initialize parameters
T_values = [10.0, 50.0, 100.0, 2000.0, 8000.0]
dt = 1.0
v = 0.05 #[0.0001, 0.05, 0.01 , 0.1]#, 0.5, 0.9]

plt.figure(1)

# for all different values of v:
for T in T_values:
    x_optimal = [0.0]
    u_optimal = []
    for t in range(int(T / dt)-1):

        # compute optimal control:
        u_optimal.append((1.0 / (T - t)) * (np.tanh(x_optimal[-1] / (v * (T - t))) - x_optimal[-1]))

        # take optimal action u:
        x_optimal.append(x_optimal[-1] + u_optimal[-1] + np.random.normal(loc=0, scale=v * dt))

    plt.plot(range(int(T / dt)), x_optimal, label='T = ' + str(T * 0.01))

plt.title('Optimal walks for different values of T')
plt.xlabel('t')
plt.ylabel('x')
plt.legend()
locks, labels = plt.xticks()
plt.xticks(locks, locks*0.01)
plt.xlim([0.0, 8000.0])
plt.show()
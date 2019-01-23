import numpy as np
import matplotlib.pyplot as plt

# initialize parameters:
T = 20.0
dt = 0.01
v = 0.01
trials = 20

# random walk:
plt.figure(0)
for n in range(trials):
    x = [0.0]
    for t in range(int(T / dt)-1):
        x.append(x[-1] + np.random.uniform(-1.0, 1.0) * dt + np.random.normal(loc=0, scale=v * dt))

    # visualize random walk:
    plt.plot(range(int(T / dt)), x)

# optimal sequence of actions:
x_optimal = [0.0]
u_optimal = []
T = T / dt
dt = 1.0
for t in range(int(T / dt)-1):
    # compute optimal control:
    u_optimal.append((1.0 / (T - t)) * (np.tanh(x_optimal[-1] / (v * (T - t))) - x_optimal[-1]))

    # take optimal action u:
    x_optimal.append(x_optimal[-1] + u_optimal[-1] + np.random.normal(loc=0, scale=v * dt))

plt.plot(range(int(T / dt)), x_optimal, linewidth=3, color='black')

# add title and labels:
plt.title('Controlled random walk')
plt.xlabel('t')
plt.ylabel('x')
locks, labels = plt.xticks()
plt.xticks(locks, locks*0.01)
plt.xlim([0.0, 2000.0])
plt.show()
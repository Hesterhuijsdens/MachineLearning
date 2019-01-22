import numpy as np
import matplotlib.pyplot as plt

# initialize parameters:
T = 20.0
dt = 0.01
v = 0.5
trials = 20

# random walk:
plt.figure(0)
for n in range(trials):
    x = [0.0]
    for t in range(int(T / dt)-1):
        x.append(x[-1] + np.random.uniform(-1.0, 1.0) * dt + np.random.normal(loc=0, scale=v * dt))

    # visualize random walk:
    plt.plot(range(int(T / dt)), x)

# add title and labels:
plt.title('Controlled random walk')
plt.xlabel('t')
plt.ylabel('x')
locks, labels = plt.xticks()
plt.xticks(locks, locks*dt)
plt.xlim([0.0, 2000.0])
plt.show()
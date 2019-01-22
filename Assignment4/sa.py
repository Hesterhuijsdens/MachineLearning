import matplotlib.pyplot as plt
from E import *


def method_sa(n, w, n_restart, nh_size):

    local_minima = []
    plt.figure()
    for t in range(n_restart):

        # initialise
        x = 2 * (np.random.uniform(0, 1, n) > 0.5).astype(int) - 1

        # stores vars at each T
        E_mean = [meanE(x, w)]
        E_std = [1, 2]

        # initialise T
        max_dE = np.inf
        if nh_size == 1:
            # estimate max dE in single spin flip
            # max_dE = state_transition(x, n, w, 0.0000001)
            max_dE = 1

        elif nh_size == 2:
            # estimate max dE in pair spin flip
            max_dE = 1

        # set variables
        # beta_init = 1.0/max_dE
        beta_init = 0.0091
        T1 = 500
        factor = 1.05
        beta = beta_init
        t2 = 0

        # Metropolitan Hastings
        while abs(E_std[t2] - E_std[t2-1]) > 2e-8:
            t2 += 1
            beta = beta * factor
            E_all = []

            # pick random neighbor
            for t1 in range(T1):
                if nh_size == 1:
                    # choose new x by flipping one random bit i
                    E2 = state_transition(x, n, w, beta)

                elif nh_size == 2:
                    # choose new x by flipping random bits i,j
                    # perform Metropolis Hasting step
                    pass

                # 2000 new states (preferably lower)
                E_all.append(E2)

            # observe convergence
            # E_mean.append((E_all)[len(E_all)-1])
            E_mean.append(np.mean(E_all))
            E_std.append(np.std(E_all))

            if t2 % 10 == 1:
                print t2, beta, E_mean[t2], E_std[t2]

        local_minima.append(E_all[len(E_all)-1])
        plt.subplot(5, 2, t+1)
        plt.plot(np.linspace(0, len(E_mean)-1, len(E_mean)-1), E_mean[1:], label="mean E")
        plt.plot(np.linspace(0, len(E_std) - 2, len(E_std) - 2), E_std[2:], label="std E")
        plt.title("SA on Ising Model, Emin = %f" % local_minima[t])
        plt.xlabel("iterations")
        plt.ylabel("E(x)")
        plt.legend()
    plt.show()
    return local_minima



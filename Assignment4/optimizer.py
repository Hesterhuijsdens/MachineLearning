from make_data import *


# minimize E= 0.5 x^T w x, with x=x_1,...,x_n and x_i=0,1
# w is a symmetric real n x n matrix with zero diagonal

def E(x, w):
    return -0.5 * x * w * x.T


method = 'sa'
nh_size = 1

# Question: what is neighboorhood size exactly? Draw.

n_restart = 1

n = 100
p = 1.0/n
w = set_w(n, p)


def method_iter():
    E_min = 1000
    for t in range(n_restart):

        # Question: do I need to wait for convergence? It is a whole search right?

        # initialise
        x = 2 * (np.random.uniform(0, 1, n) > 0.5).astype(int) - 1
        E1 = [np.sum(E(x, w)) / len(x)]
        flag = 1
        while flag == 1:
            flag = 0
            if nh_size == 1:
                # choose new x by flipping one bit i
                for i in range(len(x)):
                    x_new = x.copy()
                    x_new[i] = -x[i]

                    # compare E_new with E_old
                    if (np.sum(E(x_new, w)) / len(x_new)) < (np.sum(E(x, w)) / len(x)):
                        x = x_new
                    E1.append(np.sum(E(x, w)) / len(x))

            elif nh_size == 2:
                # choose new x by flipping bits i,j
                for i in range(len(x)):
                    x_new_i = x.copy()
                    x_new_i[i] = -x[i]
                    for j in range(len(x)):
                        x_new_j = x_new_i.copy()
                        x_new_j[j] = -x_new_i[j]
                        # compare E_ij_new with E_ij_old
                        if (np.sum(E(x_new_j, w)) / len(x_new_j)) < (np.sum(E(x_new_i, w)) / len(x_new_i)):
                            x_new_i = x_new_j
                        # E1.append(np.sum(E(x_new_i, w)) / len(x_new_i))
                    # compare E_i_new with E_i_old
                    if (np.sum(E(x_new_i, w)) / len(x_new_i)) < (np.sum(E(x, w)) / len(x)):
                        x = x_new_i
                    E1.append(np.sum(E(x, w)) / len(x))

        plt.figure()
        plt.plot(np.linspace(0, len(E1), len(E1)), E1)
        plt.title("Iterative Improvement on the Ising Model")
        plt.xlabel("iterations")
        plt.ylabel("E(x)")
        plt.show()


def method_sa():
    # initialise
    x = 2 * (np.random.uniform(0, 1, n) > 0.5).astype(int) - 1
    E1 = E(x, w)

    # stores mean E at each T
    E_outer = np.zeros((1, 100))

    # stores std E at each T
    E_bar = np.zeros((1, 100))

    # initialise T
    max_dE = 0
    if nh_size == 1:
        # estimate max dE in single spin flip: TO DO
        max_dE = 10
    elif nh_size == 2:
        # estimate max dE in pair spin flip: TO DO
        max_dE = 10

    # set initial T
    beta_init = 1/max_dE

    # length markov chain at fixed T
    T1 = 1000

    # increment of beta at each new chain
    factor = 1.05

    beta = beta_init
    E_bar[:, 1] = 1
    t2 = 1
    while E_bar[:, t2] > 0:
        t2 += 1
        beta = beta * factor
        E_all = np.zeros((1, T1))

        for t1 in range(T1):
            if nh_size == 1:
                # choose new x by flipping one random bit i
                # perform Metropolis Hasting step
                pass
            elif nh_size == 2:
                # choose new x by flipping random bits i,j
                # perform Metropolis Hasting step
                pass

            # E1 is E of new state
        #     E_all[0, t1] = E1
        #
        # E_outer[0, t2] = np.mean(E_all)
        # E_bar[0, t2] = np.std(E_all)

        # observe convergence
        # print t2, beta, E_outer[t2], E_bar[1]

    # minimal E
    # E_min = E_all[0, 1]
    # plt.plot(np.linspace(0, t2, 100), E_outer[:t2].T, label="mean E")
    # plt.plot(np.linspace(0, t2, 100), E_bar[:t2].T, label="std E")


({
    'iter': method_iter,
    'sa': method_sa
}[method])()


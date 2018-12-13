from make_data import *


# minimize E= 0.5 x^T w x, with x=x_1,...,x_n and x_i=0,1
# w is a symmetric real n x n matrix with zero diagonal

def E(x, w):
    return -0.5 * x * w * x.T


method = 'iter'
nh_size = 1
n_restart = 10

n = 100
p = 1.0/n
w = set_w(n, p)


def method_iter():
    E_min = 1000
    for t in range(n_restart):

        # initialise
        x = 2 * (np.random.uniform(0, 1, n) > 0.5).astype(int) - 1
        E1 = E(x, w)
        flag = 1

        while flag == 1:
            flag = 0
            if nh_size == 1:
                pass
                # choose new x by flipping one bit i
                # compute dE directly instead of subtracting E's of
                # different states because of efficiency
            elif nh_size == 2:
                pass
                # choose new x by flipping bits i,j

        for i in range(n):
            if x[i] == 1:
                pass
                # plt.plot(z(i, 1), z(i, 2), 'r*')
            else:
                pass
                # plt.plot(z(i, 1), z(i, 2), 'b*')

        E_min = np.minimum(E_min, E1)


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
        # estimate max dE in single spin flip
        pass
    elif nh_size == 2:
        # estimate max dE in pair spin flip
        pass

    # set initial T
    beta_init = 1/max_dE

    # length markov chain at fixed T
    T1 = 1000

    # increment of beta at each new chain
    factor = 1.05

    beta = beta_init
    E_bar[1] = 1
    t2 = 1
    while E_bar[t2] > 0:
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
            E_all[t1] = E1
        E_outer[t2] = np.mean(E_all)
        E_bar[t2] = np.std(E_all)

        # observe convergence
        print t2, beta, E_outer[t2], E_bar[1]

    # minimal E
    E_min = E_all[1]
    plt.plot(range(0, t2), E_outer[:t2])
    plt.plot(range(0, t2), E_bar[:t2])

({
    'iter': method_iter,
    'sa': method_sa
}[method])()

plt.show()

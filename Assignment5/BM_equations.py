import numpy as np


def clamped_stats(X):
    """ compute stats for all patterns at once """
    n, p = X.shape
    s = 1./p * np.sum(X, axis=1)
    ss = np.zeros((n, n))
    for xi in range(p):
        for i in range(n):
            for j in range(n):
                ss[i, j] += 1./p * X[i, xi] * X[j, xi]
    np.fill_diagonal(ss, 0)
    return s, ss


def E(index, x, w, b):
    """ compute energy for a particular state """
    s = x[index]
    n = x.shape[0]
    ss = []
    for i in range(n):
        ss.append(s * x[i])
    sw = 0.5 * np.dot(w[index], ss)
    sb = np.dot(b[index], s)
    return sw + sb


def prob_s(X, w, b, pi, state):
    """ states of low energy have high probs """
    n, p = X.shape

    Z = 0
    for i in range(n):
        Z += np.exp(E(i, X[:, pi], w, b))
    return 1./Z * (np.exp(E(state, X[:, pi], w, b)))


def free_stats(X, w, b, pi, state):
    """ this is per state within a pattern"""
    n, p = X.shape

    s = 0
    for i in range(n):
        s += X[state, pi] * prob_s(X, w, b, pi=pi, state=i)
    # print s

    ss = 0
    for i in range(n):
        for j in range(n):
            ss += X[state, pi] * X[i, pi] * prob_s(X, w, b, pi=pi, state=j)
    return s, ss


def df(s_c, ss_c, s, ss):
    db = s_c - s
    dw = ss_c - ss
    np.fill_diagonal(dw, 0)
    return db, dw


def L(X, w, b):
    n, p = X.shape
    ps = 0

    # ps for all patterns
    for pi in range(p):

        # ps for all states in pattern
        ps_u = 1
        for si in range(n):
            ps_u = np.multiply(ps_u, prob_s(X, w, b, pi=pi, state=si))
        ps += np.log(ps_u)

    # print 1./p * ps
    return 1./p * ps


def train_BM(X, eta=0.01, n_epochs=200):
    n, p = X.shape

    # random weights initialisation
    w = np.loadtxt('w.txt')

    # random threshold initialisation
    # b = np.loadtxt('b.txt')
    b = np.zeros(n)

    # expectations under empirical distribution (on training patterns)
    dE_dw, dE_db = compute_expectations(X)

    # compute clamped statistics
    # s_c, ss_c = clamped_stats(X)

    # E(0, X[:, 2], w, b)
#
    # L(X, w, b)

    # gibbs_sampling(w, b, n_gibbs=20, n_burnin=10)

    for i in range(0, 50):

        # compute free statistics
        # s, ss = free_stats(X, w, b, pi=2, state=0)
        # Gibbs sampling with current model
        XM = gibbs_sampling(w, b)

        # compute gradients
        # db, dw = df(s_c, ss_c, s, ss)
        dEM_dw, dEM_db = compute_expectations(XM)


        # weight update
        # w += eta * dw
        # b += eta * db

        # update weights and biases
        w = w + (eta * (dEM_dw - dE_dw))
        b = b + (eta * (dEM_db - dE_db))

        # L(X, w, b)

    return w, b






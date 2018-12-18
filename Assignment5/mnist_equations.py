from binary_data import *
from itertools import *


def flip_prob(w, x, b, j):
    return np.exp(2 * (np.dot(w, x) + b) * x[j])


def E(x, w, b):
    """ compute energy for a particular state """
    n = len(x)
    E = 0
    for i in range(n):
        for j in range(n):
            if i is not j:
                E += 0.5 * w[i, j] * x[i] * x[j] + b[i] * x[i]
    return E


def F(wij, m, b):

    return 1


def state_prob(X, w, b, pi):
    """ states of low energy should have high probs """
    F(w, m, b)
    return 1./F * (np.exp(-E(X[:, pi], w, b)))



def gibbs_sampling(w, b, n_gibbs=500, n_burnin=10):
    """ approximate model distribution for training a BM """
    # 10 nodes
    n_nodes = w.shape[0]

    # array for saving node states for each time step
    X = np.zeros((n_nodes, n_gibbs))

    # state vector initialisation (t=0)
    X[:, 0] = np.random.randint(2, size=n_nodes)
    for i in range(n_nodes):
        if X[i, 0] >= 0.5:
            X[i, 0] = 1.0
        else:
            X[i, 0] = -1.0

    # loop over Gibbs samples
    for i in range(1, n_gibbs):

        # loop over nodes
        for j in range(n_nodes):

            # compute flip probability
            p = flip_prob(w[:, j], X[:, i-1], b[j], j)

            # determine new binary state
            if (np.random.rand() < 1./p).astype("float"):
                X[j, i] = -X[j, i-1]
            else:
                X[j, i] = X[j, i-1]

    # discard burn-in (depend on state initialisation)
    return X[:, n_burnin:]


# Compute expectations
def compute_expectations(X):
    """ compute the expectation (mean over patterns / samples) of the
    partial derivatives for w and b"""

    # 1 pattern is training example of length m
    dw = (np.dot(X, X.T)) / X.shape[1]
    np.fill_diagonal(dw, 0)
    db = np.mean(X, axis=1)
    return dw, db


# def log_likelihood(X, w, b):
#     n, p = np.shape(X)
#     ps = np.zeros((p, n))
#
#     # loop over patterns
#     for pi in range(p):
#         ps[pi] = state_prob(X, w, b, pi)
#     return np.mean(np.sum(np.log(ps), axis=1))


def boltzmann_train(patterns, eta, n_epochs=2000, n_gibbs=500, n_burnin=10):
    n_nodes, n_examples = np.shape(patterns)

    # weights initialisation
    # w = np.loadtxt('w.txt')
    w = get_w(10)
    w_list = np.zeros((n_epochs, n_nodes, n_nodes))

    # bias initialisation
    b = np.zeros(n_nodes)

    # E(patterns[:, 2], w, b)
    # state_prob(patterns, w, b, 2)
    # print log_likelihood(patterns, w, b)

    # expectations under empirical distribution (training patterns)
    dE_dw, dE_db = compute_expectations(patterns)

    # print E(3, patterns[:, 2], w, b), state_prob(patterns, w, b, 2, 1)

    # loop over epochs
    for i_epoch in range(n_epochs):

        # Gibbs sampling with current model: free stats
        XM = gibbs_sampling(w, b, n_gibbs, n_burnin)

        # expectations under model distribution:
        dEM_dw, dEM_db = compute_expectations(XM)

        # update weights and biases
        w += (eta * (dEM_dw - dE_dw))
        b += (eta * (dEM_db - dE_db))
        w_list[i_epoch, :, :] = w

        # E should go down, prob should go up
        # print E(patterns[:, 0], w, b), state_prob(patterns, w, b, 0)

    # force symmetry
    w = (w + w.T) / 2
    # print log_likelihood(patterns, w, b)
    return w, b, w_list


# Boltzmann dreaming
def boltzmann_dream(w, b, n_epochs=20):
    return gibbs_sampling(w, b, n_gibbs=20, n_burnin=10)

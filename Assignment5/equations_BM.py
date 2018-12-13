import numpy as np
import matplotlib.pyplot as plt


def sigmoid(A):
    return 1 / (1 + np.exp(-A))


def flip_prob(w, x, b):
    """ probability """
    return sigmoid(1.0 * (np.dot(w, x) + b))


def E(index, x, w, b):
    """ compute energy for a particular state """
    sw = 0.5 * np.dot(w[index], np.dot(x[index], x))
    sb = np.dot(b[index], x[index])
    return sw + sb


def state_prob(X, w, b, pi, state):
    """ states of low energy have high probs """
    n, p = X.shape

    Z = 0
    for i in range(n):
        Z += np.exp(E(i, X[:, pi], w, b))
    return 1./Z * (np.exp(E(state, X[:, pi], w, b)))


def gibbs_sampling(w, b, n_gibbs, n_burnin):
    """ approximate model distribution for training a BM """
    # 10 nodes
    n_nodes = w.shape[0]

    # array for saving node states for each time step
    X = np.zeros((n_nodes, n_gibbs))

    # state vector initialisation (t=0)
    X[:, 0] = np.random.randint(2, size=n_nodes)

    # loop over Gibbs samples
    for i in range(1, n_gibbs):

        # loop over nodes
        for j in range(n_nodes):

            # compute flip probability
            p = flip_prob(w[:, j], X[:, i-1], b[j])

            # determine new binary state
            X[j, i] = (np.random.rand() < p).astype("float")

    # discard burn-in (depend on state initialisation)
    return X[:, n_burnin:]


# Compute expectations
def compute_expectations(X):
    """ compute the expectation (mean over patterns / samples) of the
    partial derivatives for w and b"""

    # 1 pattern is training example of length m
    dw = (-np.dot(X, X.T)) / X.shape[1]
    np.fill_diagonal(dw, 0)
    db = -np.mean(X, axis=1)
    return dw, db


def log_likelihood(X, w, b):
    n, p = X.shape
    ps = np.zeros((p, n))

    # loop over patterns
    for pi in range(p):

        # loop over states
        for si in range(n):
            ps[pi, si] = state_prob(X, w, b, pi, si)

    # print ps
    # print np.prod(ps, axis=1),
    # print np.mean(np.prod(ps, axis=1))
    print " "

    # print np.sum(np.log(ps), axis=1)
    print np.mean(np.sum(np.log(ps), axis=1))


def boltzmann_train(patterns, eta=0.01, n_epochs=200, n_gibbs=500, n_burnin=10):
    n_nodes, n_examples = patterns.shape

    # weights initialisation
    w = np.loadtxt('w.txt')

    # bias initialisation
    b = np.zeros(n_nodes)
    # b = np.loadtxt('b.txt')

    # print E(1, patterns[:, 2], w, b), state_prob(patterns, w, b, 2, 1)
    log_likelihood(patterns, w, b)

    # expectations under empirical distribution (training patterns)
    dE_dw, dE_db = compute_expectations(patterns)

    # E(3, patterns[:, 2], w, b)
    # state_prob(patterns, w, b, 2, 1)

    # loop over epochs
    for i_epoch in range(n_epochs):
        # print("Epoch {}/{}.".format(1 + i_epoch, n_epochs))

        # Gibbs sampling with current model: free stats
        XM = gibbs_sampling(w, b, n_gibbs, n_burnin)
        # print "Gibbs done"

        # expectations under model distribution:
        dEM_dw, dEM_db = compute_expectations(XM)
        # print "CE done"
        # print "start update"

        # update weights and biases
        w = w + (eta * (dEM_dw - dE_dw))
        b = b + (eta * (dEM_db - dE_db))

        # if i_epoch == n_epochs-1:
        #     print w, b

    # force symmetry
    w = (w + w.T) / 2

    # print E(1, patterns[:, 2], w, b), state_prob(patterns, w, b, 2, 1)
    print ""
    log_likelihood(patterns, w, b)
    # print("Training done.")
    return w, b
#
#
# # Boltzmann dreaming
# def boltzmann_dream(w, b, n_epochs=20):
#
#     # sample 60,000 images
#     return gibbs_sampling(w, b, n_gibbs=20, n_burnin=10)
#
#
# train BM
patterns = np.loadtxt('data.txt')
w, b = boltzmann_train(patterns, n_epochs=200)
#
# # test BM
# X_sample = boltzmann_dream(w, b)
# plt.figure()
# plt.imshow(X_sample)
# plt.show()


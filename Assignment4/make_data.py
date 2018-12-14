import numpy as np
import scipy.stats as stats
import scipy.sparse as sparse
import matplotlib.pyplot as plt

np.random.seed(5)


def sprandsym(n, density):
    rvs = stats.norm().rvs
    X = sparse.random(n, n, density=density, data_rvs=rvs)
    upper_X = sparse.triu(X)
    result = upper_X + upper_X.T - sparse.diags(X.diagonal())
    return result


def set_w(n, p):
    w = sprandsym(n, p)

    # frustrated systems
    # w = (w > 0).astype(int) - (w < 0).astype(int)

    # ferro-magnetic (easy) systems
    w = (w > 0).astype(int)

    # diagonal elements to zero
    x_lil = sparse.lil_matrix(w)
    x_lil.setdiag(0)
    return x_lil.tocsr()



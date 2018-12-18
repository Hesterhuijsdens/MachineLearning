from make_data import *
from sa import *
from iter import *
from scipy.sparse import csr_matrix


# minimize E= 0.5 x^T w x, with x=x_1,...,x_n and x_i=0,1
# w is a symmetric real n x n matrix with zero diagonal


method = 'iter'
nh_size = 1

# Question: what is neighboorhood size exactly? Draw.

n_restart = 10
n = 50
p = 1.0/n
w = set_w(n, p)

lm_sa = method_sa(n, w, n_restart, nh_size)
# lm_iter = method_iter(n, w, n_restart, nh_size)
print "sa: ", lm_sa
# print "iter: ", lm_iter

# iter is op elk vlak beter???




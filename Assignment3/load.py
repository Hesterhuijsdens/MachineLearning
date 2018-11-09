import tarfile
import numpy as np


def loadLasso(version="train"):
    files = tarfile.open('lasso_data.tar', mode='r').extractall()

    if version == "train":
        input = open('lasso_data/data1_input_train', 'r').read().split("  ")
        output = open('lasso_data/data1_output_train', 'r').read().split("  ")
    else:
        input = open('lasso_data/data1_input_val', 'r').read().split("  ")
        output = open('lasso_data/data1_output_val', 'r').read().split("  ")

    input = ' '.join(input).replace('\n', '').split()
    output = ' '.join(output).replace('\n', '').split()
    return np.reshape(input, (50, 100)).astype(np.float), np.reshape(output, (50,)).astype(np.float)


a, b = loadLasso()
print np.shape(a)
print np.shape(b)

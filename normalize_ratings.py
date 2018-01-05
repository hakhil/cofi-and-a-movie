import numpy as np

def normalize(Y):
    m, n = Y.shape
    Ymean = np.zeros([m, 1])
    Ynorm = np.zeros([m, n])
    for i in range(m):
        if np.count_nonzero(Y[i, :]) > 0:
            Ymean[i] = sum(Y[i, :]) / np.count_nonzero(Y[i, :])
            for j in range(n):
                if Y[i, j] > 0:
                    Ynorm[i, j] = Y[i, j] - Ymean[i]
    return Ymean, Ynorm
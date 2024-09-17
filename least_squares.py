import numpy as np

def least_squares(X, y):
    M = X.T @ X
    s = X.T @ y
    return np.linalg.solve(M,s )
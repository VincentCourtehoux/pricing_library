import numpy as np
from numpy.polynomial.laguerre import lagval

def laguerre_matrix(x, degree):
    """
    Compute the matrix of Laguerre polynomials evaluated at points x.

    Parameters:
    x (array-like): 1D array of points where polynomials are evaluated.
    degree (int): Maximum degree of Laguerre polynomials.

    Returns:
    mat (ndarray): Matrix of shape (len(x), degree+1) where each column i contains values of the i-th Laguerre polynomial at points x.
    """
    mat = np.zeros((len(x), degree+1))
    for i in range(degree+1):
        c = np.zeros(i+1)
        c[-1] = 1
        mat[:, i] = lagval(x, c)
    return mat
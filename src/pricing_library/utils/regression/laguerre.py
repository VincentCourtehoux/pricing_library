import numpy as np
from numpy.polynomial.laguerre import lagval
from .base_regression import BaseRegression

class LaguerreRegression(BaseRegression):
    def __init__(self, degree=2):
        if degree < 0:
            raise ValueError("Degree must be non-negative")
        self.degree = degree + 1
        self.coeffs = [np.zeros(i + 1) for i in range(self.degree)]
        for i, c in enumerate(self.coeffs):
            c[-1] = 1
    
    def _create_laguerre_basis(self, x):
        
        n = len(x)
        basis = np.zeros((n, self.degree))
        
        for i in range(self.degree):
            basis[:, i] = lagval(x, self.coeffs[i])
        
        return basis
    
    def fit_predict(self, X, y, X_pred):
        
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
            
        if len(X) == 0:
            return np.zeros_like(X_pred)
            
        X_flat = X.flatten()
        X_pred_flat = X_pred.flatten()
        
        X_design = self._create_laguerre_basis(X_flat)
        X_pred_design = self._create_laguerre_basis(X_pred_flat)

        beta, residuals, rank, s = np.linalg.lstsq(X_design, y, rcond=None)
        
        return X_pred_design @ beta
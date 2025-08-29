import numpy as np
from .base_regression import BaseRegression

class PolynomialRegression(BaseRegression):
    def __init__(self, degree=2):
        if degree < 0:
            raise ValueError("Degree must be non-negative")
        self.degree = degree
    
    def fit_predict(self, X, y, X_pred):
        
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
            
        if len(X) == 0:
            return np.zeros_like(X_pred)
            
        X_flat = X.flatten()
        X_pred_flat = X_pred.flatten()
        
        X_design = np.vander(X_flat, N=self.degree + 1, increasing=True)
        X_pred_design = np.vander(X_pred_flat, N=self.degree + 1, increasing=True)

        beta, residuals, rank, s = np.linalg.lstsq(X_design, y, rcond=None)
        
        return X_pred_design @ beta
from .base_regression import BaseRegression
from .polynomial import PolynomialRegression
from .laguerre import LaguerreRegression

def get_regressor(regression_type='polynomial', **kwargs):
    if regression_type == 'polynomial':
        return PolynomialRegression(**kwargs)
    elif regression_type == 'laguerre':
        return LaguerreRegression(**kwargs)
    else:
        raise ValueError(f"Unknown regression type: {regression_type}")

__all__ = [
    'BaseRegression',
    'PolynomialRegression', 
    'LaguerreRegression',
    'get_regressor'
]
from .base_calculator import BaseCalculator
from ..models.least_squares_mc import LeastSquaresMC
from ..models.binomial import BinomialModel

class AmericanCalculator(BaseCalculator):
    def __init__(self, models=None):
        self.models = models or {
            'least_squares_mc': LeastSquaresMC(),
            'binomial_tree': BinomialModel()
        }

    def calculate(self, params, method='least_squares_mc'):
        if method not in self.models:
            raise ValueError(f'Unsupported method: {method}')
        
        model = self.models[method]
        base_params = self._extract_base_params(params)
        extra_params = self._extract_extra_params(params, method)
        
        return model.calculate(**base_params, **extra_params)
    
    def calculate_greeks(self, params, method='least_squares_mc'):
        if method not in self.models:
            raise ValueError(f'Unsupported method: {method}')
        
        model = self.models[method]
        base_params = self._extract_base_params(params)
        extra_params = self._extract_extra_params(params, method)
        
        return model.calculate_greeks(**base_params, **extra_params)
    
    def _extract_base_params(self, params):
        return {
            'S': params['S'],
            'K': params['K'], 
            'T': params['T'],
            'r': params['r'],
            'sigma': params['sigma'],
            'option_type': params['option_type'],
            'q': params.get('q', 0)
        }
    
    def _extract_extra_params(self, params, method):
        extra_params = {
            'option_style': 'american'
        }
        
        if method == 'least_squares_mc':
            extra_params.update({
                'n_paths': params.get('monte_carlo_paths', 10000),
                'n_steps': params.get('monte_carlo_steps', 100),
                'regression_type': params.get('regression_type', 'polynomial'),
                'regression_degree': params.get('regression_degree', 2),
                'seed': params.get('seed', None)
            })
        elif method == 'binomial_tree':
            extra_params.update({
                'n_steps': params.get('binomial_steps', 100)
            })
        
        return extra_params
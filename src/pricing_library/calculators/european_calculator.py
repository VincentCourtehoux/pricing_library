import numpy as np
from ..models.black_scholes import BlackScholesModel
from ..models.monte_carlo import VanillaMonteCarlo
from ..models.binomial import BinomialModel
from .base_calculator import BaseCalculator

class EuropeanCalculator(BaseCalculator):
    def __init__(self, models=None):
        self.models = models or {
            'black_scholes': BlackScholesModel(),
            'monte_carlo': VanillaMonteCarlo(),
            'binomial_tree': BinomialModel()
        }

    def calculate(self, params, method='black_scholes'):
        if method not in self.models:
            raise ValueError(f'Unsupported method: {method}')
        
        model = self.models[method]
        base_params = self._extract_base_params(params)
        extra_params = self._extract_extra_params(params, method)

        if any(isinstance(v, np.ndarray) for v in base_params.values()):
            return self._vectorized_calculate(model, base_params, extra_params)
        else:
            return model.calculate(**base_params, **extra_params)
    
    def calculate_greeks(self, params, method='black_scholes'):
        if method not in self.models:
            raise ValueError(f'Unsupported method: {method}')
        
        model = self.models[method]
        base_params = self._extract_base_params(params)
        extra_params = self._extract_extra_params(params, method)

        if any(isinstance(v, np.ndarray) for v in base_params.values()):
            return self._vectorized_greeks(model, base_params, extra_params)
        else:
            return model.calculate_greeks(**base_params, **extra_params)

    def _vectorized_calculate(self, model, base_params, extra_params):
        arrays = {k: np.atleast_1d(v) for k, v in base_params.items()}
        n = max(len(v) for v in arrays.values())
        for k, v in arrays.items():
            if len(v) == 1:
                arrays[k] = np.full(n, v[0])
        
        prices = model.calculate(**arrays, **extra_params)['price']
        return {'price': np.array(prices), 'method': model.__class__.__name__}

    def _vectorized_greeks(self, model, base_params, extra_params):
        arrays = {k: np.atleast_1d(v) for k, v in base_params.items()}
        n = max(len(v) for v in arrays.values())
        for k, v in arrays.items():
            if len(v) == 1:
                arrays[k] = np.full(n, v[0])

        greeks = model.calculate_greeks(**arrays, **extra_params)
        return {k: np.array(v) for k, v in greeks.items()}

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
        extra_params = {}
        if method == 'monte_carlo':
            extra_params.update({
                'n_paths': params.get('monte_carlo_paths', 10000),
                'n_steps': params.get('monte_carlo_steps', 100),
                'option_style': 'european',
                'seed': params.get('seed', None)
            })
        elif method == 'binomial_tree':
            extra_params.update({
                'n_steps': params.get('binomial_steps', 100),
                'option_style': 'european'
            })
        return extra_params

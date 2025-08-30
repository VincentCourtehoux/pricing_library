from .base_calculator import BaseCalculator
from ..models.black_scholes_barrier import BlackScholesBarrierModel
from ..models.monte_carlo import VanillaMonteCarlo

class BarrierCalculator(BaseCalculator):
    def __init__(self, models=None):
        self.models = models or {
            'black_scholes': BlackScholesBarrierModel(),
            'monte_carlo': VanillaMonteCarlo()
        }

    def calculate(self, params, method='black_scholes'):
        if method not in self.models:
            raise ValueError(f'Unsupported method: {method}')
        
        model = self.models[method]
        base_params = self._extract_base_params(params)
        extra_params = self._extract_extra_params(params, method)
        
        return model.calculate(**base_params, **extra_params)
    
    def calculate_greeks(self, params, method='black_scholes'):
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
            'barrier_type': params['barrier_type'],
            'barrier_level': params['barrier_level']
        }
        
        if method == 'monte_carlo':
            extra_params.update({
                'n_paths': params.get('monte_carlo_paths', 10000),
                'n_steps': params.get('monte_carlo_steps', 100),
                'seed': params.get('seed', None)
            })
        
        return extra_params
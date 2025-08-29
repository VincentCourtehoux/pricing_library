from .base_calculator import BaseCalculator
from ..models.monte_carlo import VanillaMonteCarlo

class AsianCalculator(BaseCalculator):
    def __init__(self, models=None):
        self.models = models or {
            'monte_carlo': VanillaMonteCarlo(),
        }

    def calculate(self, params, method='monte_carlo'):
        if method not in self.models:
            raise ValueError(f'Unsupported method: {method}')
        
        model = self.models[method]
        base_params = self._extract_base_params(params)
        extra_params = self._extract_extra_params(params, method)
        
        return model.calculate(**base_params, **extra_params)
    
    def calculate_greeks(self, params, method='monte_carlo'):
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
            'option_style': 'asian',
            'averaging_type': params.get('averaging_type', 'arithmetic'),
            'monitoring_dates': params.get('monitoring_dates', None)
        }
        
        if method == 'monte_carlo':
            extra_params.update({
                'n_paths': params.get('monte_carlo_paths', 10000),
                'n_steps': params.get('monte_carlo_steps', 100)
            })
        
        return extra_params
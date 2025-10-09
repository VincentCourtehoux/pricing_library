import numpy as np
from ..models.black_scholes import BlackScholesModel
from ..models.monte_carlo import VanillaMonteCarlo
from ..models.binomial import BinomialModel
from .base_calculator import BaseCalculator

class OptionCombinationCalculator(BaseCalculator):
    def __init__(self, models=None):
        self.models = models or {
            'black_scholes': BlackScholesModel(),
            'monte_carlo': VanillaMonteCarlo(),
            'binomial_tree': BinomialModel()
        }

        self._combination_methods = ['straddle', 'strangle', 'bull_spread', 'bear_spread']

    def calculate(self, params, method='black_scholes'):
        if method not in self.models:
            raise ValueError(f'Unsupported method: {method}')

        combination_type = params.get('combination')
        if combination_type not in self._combination_methods:
            raise ValueError(
                f"Unsupported combination type: {combination_type}. Supported: {self._combination_methods}"
            )

        model = self.models[method]
        base_params = self._extract_base_params(params, combination_type)
        extra_params = self._extract_extra_params(params, method)

        if combination_type in ['straddle', 'strangle']:
            call_params = base_params.copy()
            call_params['option_type'] = np.full_like(base_params['S'], 'call', dtype=object)
            if combination_type == 'strangle':
                call_params['K'] = base_params['K_call']

            put_params = base_params.copy()
            put_params['option_type'] = np.full_like(base_params['S'], 'put', dtype=object)
            if combination_type == 'strangle':
                put_params['K'] = base_params['K_put']

            call_price = model.calculate(**call_params, **extra_params)['price']
            put_price = model.calculate(**put_params, **extra_params)['price']
            total_price = call_price + put_price

            legs = [
                {'type': 'call', 'price': call_price, 'K': call_params.get('K', base_params.get('K'))},
                {'type': 'put', 'price': put_price, 'K': put_params.get('K', base_params.get('K'))}
            ]

        elif combination_type in ['bull_spread', 'bear_spread']:
            n = len(base_params['S'])
            long_params = base_params.copy()
            short_params = base_params.copy()

            if combination_type == 'bull_spread':
                long_params['option_type'] = np.full(n, 'call', dtype=object)
                short_params['option_type'] = np.full(n, 'call', dtype=object)
                long_params['K'] = base_params['K1']
                short_params['K'] = base_params['K2']
            else:  
                long_params['option_type'] = np.full(n, 'put', dtype=object)
                short_params['option_type'] = np.full(n, 'put', dtype=object)
                long_params['K'] = base_params['K2']
                short_params['K'] = base_params['K1']

            long_price = model.calculate(**long_params, **extra_params)['price']
            short_price = model.calculate(**short_params, **extra_params)['price']
            total_price = long_price - short_price

            legs = [
                {'type': long_params['option_type'][0], 'price': long_price, 'K': long_params['K']},
                {'type': short_params['option_type'][0], 'price': short_price, 'K': short_params['K']}
            ]

        return {
            'price': total_price,
            'legs': legs,
            'method': method,
            'option_style': combination_type
        }

    def calculate_greeks(self, params, method='black_scholes'):
        if method not in self.models:
            raise ValueError(f'Unsupported method: {method}')

        combination_type = params.get('combination')
        if combination_type not in self._combination_methods:
            raise ValueError(
                f"Unsupported combination type: {combination_type}. Supported: {self._combination_methods}"
            )

        model = self.models[method]
        base_params = self._extract_base_params(params, combination_type)
        extra_params = self._extract_extra_params(params, method)

        def compute_combined(long_params, short_params=None, subtract=False):
            long_greeks = model.calculate_greeks(**long_params, **extra_params)
            if short_params is not None:
                short_greeks = model.calculate_greeks(**short_params, **extra_params)
                combined_greeks = {k: long_greeks[k] - short_greeks[k] if subtract else long_greeks[k] + short_greeks[k]
                                for k in long_greeks}
                combined_price = long_greeks['price'] - short_greeks['price'] if subtract else long_greeks['price'] + short_greeks['price']
            else:
                combined_greeks = long_greeks
                combined_price = long_greeks['price']
            return combined_price, combined_greeks

        if combination_type in ['straddle', 'strangle']:
            call_params = base_params.copy()
            call_params['option_type'] = np.full_like(base_params['S'], 'call', dtype=object)
            if combination_type == 'strangle':
                call_params['K'] = base_params['K_call']

            put_params = base_params.copy()
            put_params['option_type'] = np.full_like(base_params['S'], 'put', dtype=object)
            if combination_type == 'strangle':
                put_params['K'] = base_params['K_put']

            total_price, combined_greeks = compute_combined(call_params, put_params)

        elif combination_type in ['bull_spread', 'bear_spread']:
            n = len(base_params['S'])
            long_params = base_params.copy()
            short_params = base_params.copy()

            if combination_type == 'bull_spread':
                long_params['option_type'] = np.full(n, 'call', dtype=object)
                short_params['option_type'] = np.full(n, 'call', dtype=object)
                long_params['K'] = base_params['K1']
                short_params['K'] = base_params['K2']
            else:
                long_params['option_type'] = np.full(n, 'put', dtype=object)
                short_params['option_type'] = np.full(n, 'put', dtype=object)
                long_params['K'] = base_params['K2']
                short_params['K'] = base_params['K1']

            total_price, combined_greeks = compute_combined(long_params, short_params, subtract=True)

        return {
            'option_style': combination_type,
            'method': method,
            'price': total_price,
            'greeks': combined_greeks
        }


    def _extract_base_params(self, params, combination_type=None):
        base = {
            'S': np.atleast_1d(params['S']),
            'T': np.atleast_1d(params['T']),
            'r': np.atleast_1d(params['r']),
            'sigma': np.atleast_1d(params['sigma']),
            'q': np.atleast_1d(params.get('q', 0))
        }

        if combination_type == 'straddle':
            base['K'] = np.atleast_1d(params['K'])
        elif combination_type == 'strangle':
            base['K_call'] = np.atleast_1d(params['K_call'])
            base['K_put'] = np.atleast_1d(params['K_put'])
        elif combination_type in ['bull_spread', 'bear_spread']:
            base['K1'] = np.atleast_1d(params['K1'])
            base['K2'] = np.atleast_1d(params['K2'])
        else:
            base['K'] = np.atleast_1d(params.get('K'))

        return base

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

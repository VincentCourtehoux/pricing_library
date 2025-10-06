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

        self._combination_methods = {
            'straddle': self._calculate_straddle,
            'strangle': self._calculate_strangle,
            'bull_spread': self._calculate_bull_spread
        }

    def calculate(self, params, method='black_scholes'):
        if method not in self.models:
            raise ValueError(f'Unsupported method: {method}')

        combination_type = params.get('combination')
        if combination_type not in self._combination_methods:
            raise ValueError(
                f"Unsupported combination type: {combination_type}. "
                f"Supported: {list(self._combination_methods.keys())}"
            )

        return self._combination_methods[combination_type](params, method)

    def _calculate_straddle(self, params, method):
        model = self.models[method]
        base_params = self._extract_base_params(params, 'straddle')
        extra_params = self._extract_extra_params(params, method)

        call_params = base_params.copy()
        call_params['option_type'] = 'call'
        put_params = base_params.copy()
        put_params['option_type'] = 'put'

        call_price = model.calculate(**call_params, **extra_params)['price']
        put_price = model.calculate(**put_params, **extra_params)['price']

        return {
            'price': call_price + put_price,
            'legs': [
                {'type': 'call', 'price': call_price, 'K': base_params['K']},
                {'type': 'put', 'price': put_price, 'K': base_params['K']}
            ],
            'method': method,
            'option_style': 'straddle'
        }

    def _calculate_strangle(self, params, method):
        model = self.models[method]
        base_params = self._extract_base_params(params, 'strangle')
        extra_params = self._extract_extra_params(params, method)

        call_params = base_params.copy()
        call_params['option_type'] = 'call'
        call_params['K'] = base_params['K_call']
        put_params = base_params.copy()
        put_params['option_type'] = 'put'
        put_params['K'] = base_params['K_put']

        call_price = model.calculate(**call_params, **extra_params)['price']
        put_price = model.calculate(**put_params, **extra_params)['price']

        return {
            'price': call_price + put_price,
            'legs': [
                {'type': 'call', 'price': call_price, 'K': base_params['K_call']},
                {'type': 'put', 'price': put_price, 'K': base_params['K_put']}
            ],
            'method': method,
            'option_style': 'strangle'
        }

    def _calculate_bull_spread(self, params, method):
        model = self.models[method]
        base_params = self._extract_base_params(params, 'bull_spread')
        extra_params = self._extract_extra_params(params, method)

        long_params = base_params.copy()
        long_params['option_type'] = 'call'
        long_params['K'] = base_params['K1']
        short_params = base_params.copy()
        short_params['option_type'] = 'call'
        short_params['K'] = base_params['K2']

        long_price = model.calculate(**long_params, **extra_params)['price']
        short_price = model.calculate(**short_params, **extra_params)['price']

        return {
            'price': long_price - short_price,
            'legs': [
                {'type': 'call', 'price': long_price, 'K': base_params['K1']},
                {'type': 'call', 'price': short_price, 'K': base_params['K2']}
            ],
            'method': method,
            'option_style': 'bull_spread'
        }

    def _extract_base_params(self, params, combination_type=None):
        base = {
            'S': params['S'],
            'T': params['T'],
            'r': params['r'],
            'sigma': params['sigma'],
            'q': params.get('q', 0)
        }

        if combination_type == 'straddle':
            if 'K' not in params:
                raise ValueError("Parameter 'K' is required for straddle")
            base['K'] = params['K']
        elif combination_type == 'strangle':
            if 'K_call' not in params or 'K_put' not in params:
                raise ValueError("Parameters 'K_call' and 'K_put' are required for strangle")
            base['K_call'] = params['K_call']
            base['K_put'] = params['K_put']
        elif combination_type == 'bull_spread':
            if 'K1' not in params or 'K2' not in params:
                raise ValueError("Parameters 'K1' and 'K2' are required for strangle")
            base['K1'] = params['K1']
            base['K2'] = params['K2']
        else:
            base['K'] = params.get('K')

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

    def calculate_greeks(self, params, method='black_scholes'):
        if method not in self.models:
            raise ValueError(f'Unsupported method: {method}')

        combination_type = params.get('combination')
        if combination_type not in self._combination_methods:
            raise ValueError(
                f"Unsupported combination type: {combination_type}. "
                f"Supported: {list(self._combination_methods.keys())}"
            )

        model = self.models[method]
        base_params = self._extract_base_params(params, combination_type)
        extra_params = self._extract_extra_params(params, method)

        if combination_type == 'straddle':
            call_params = base_params.copy()
            call_params['option_type'] = 'call'
            put_params = base_params.copy()
            put_params['option_type'] = 'put'

            call_greeks = model.calculate_greeks(**call_params, **extra_params)
            put_greeks = model.calculate_greeks(**put_params, **extra_params)

            combined_greeks = {
                'price': call_greeks['price'] + put_greeks['price'],
                'delta': call_greeks['delta'] + put_greeks['delta'],
                'gamma': call_greeks['gamma'] + put_greeks['gamma'],
                'vega': call_greeks['vega'] + put_greeks['vega'],
                'theta': call_greeks['theta'] + put_greeks['theta'],
                'rho': call_greeks['rho'] + put_greeks['rho']
            }
        elif combination_type == 'strangle':
            call_params = base_params.copy()
            call_params['option_type'] = 'call'
            call_params['K'] = base_params['K_call'] 

            put_params = base_params.copy()
            put_params['option_type'] = 'put'
            put_params['K'] = base_params['K_put']

            call_greeks = model.calculate_greeks(**call_params, **extra_params)
            put_greeks = model.calculate_greeks(**put_params, **extra_params)

            combined_greeks = {
                'price': call_greeks['price'] + put_greeks['price'],
                'delta': call_greeks['delta'] + put_greeks['delta'],
                'gamma': call_greeks['gamma'] + put_greeks['gamma'],
                'vega': call_greeks['vega'] + put_greeks['vega'],
                'theta': call_greeks['theta'] + put_greeks['theta'],
                'rho': call_greeks['rho'] + put_greeks['rho']
            }
        
        elif combination_type == 'bull_spread':
            long_params = base_params.copy()
            long_params['option_type'] = 'call'
            long_params['K'] = base_params['K1'] 

            short_params = base_params.copy()
            short_params['option_type'] = 'call'
            short_params['K'] = base_params['K2']

            long_greeks = model.calculate_greeks(**long_params, **extra_params)
            short_greeks = model.calculate_greeks(**short_params, **extra_params)

            combined_greeks = {
                'price': long_greeks['price'] - short_greeks['price'],
                'delta': long_greeks['delta'] - short_greeks['delta'],
                'gamma': long_greeks['gamma'] - short_greeks['gamma'],
                'vega': long_greeks['vega'] - short_greeks['vega'],
                'theta': long_greeks['theta'] - short_greeks['theta'],
                'rho': long_greeks['rho'] - short_greeks['rho']
            }

        return {
            'option_style': combination_type,
            'method': method,
            'greeks': combined_greeks
        }

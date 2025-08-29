import numpy as np
from ..utils.stochastic_processes.gbm import GeometricBrownianMotion
from ..utils.payoff import intrinsic_value, asian_payoff
from .base_model import PricingModel

class VanillaMonteCarlo(PricingModel):
    def calculate(self, S, K, T, r, sigma, option_type='call', q=0.0, **kwargs):
        n_paths = kwargs.get('n_paths', 10000)
        n_steps = kwargs.get('n_steps', 100)
        option_style = kwargs.get('option_style', 'european')

        paths = GeometricBrownianMotion.simulate(S, T, r, sigma, q, n_paths, n_steps)
        final_prices = paths[:, -1]

        if option_style == 'european':
            payoffs = intrinsic_value(final_prices, K, option_type)
    
        elif option_style == 'asian':
            averaging_type = kwargs.get('averaging_type', 'arithmetic')
            monitoring_dates = kwargs.get('monitoring_dates', None)

            if monitoring_dates is None:
                monitored_prices = paths[:, :]
            else:
                time_points = np.linspace(0, T, n_steps + 1)
                indices = [np.argmin(np.abs(time_points - t)) for t in monitoring_dates]
                monitored_prices = paths[:, indices]

            payoffs = asian_payoff(monitored_prices, K, option_type, averaging_type)
        
        elif option_style == 'barrier':
            barrier_type = kwargs.get('barrier_type', None)
            barrier_level = kwargs.get('barrier_level', None)

            if barrier_type is None:
                raise ValueError("barrier_type is required for barrier options")
            if barrier_level is None:
                raise ValueError("barrier_level is required for barrier options")
            
            if barrier_type == "up-and-in":
                barrier_valid = (np.max(paths, axis=1) >= barrier_level)
            elif barrier_type == "up-and-out":
                barrier_valid = (np.max(paths, axis=1) < barrier_level)
            elif barrier_type == "down-and-in":
                barrier_valid = (np.min(paths, axis=1) <= barrier_level)
            elif barrier_type == "down-and-out":
                barrier_valid = (np.min(paths, axis=1) > barrier_level)
            else:
                raise ValueError("Invalid barrier_type")
            
            payoffs = intrinsic_value(final_prices, K, option_type) * barrier_valid
        
        elif option_style == 'gap':
            K1 = kwargs.get('K1', None)
            K2 = kwargs.get('K2', None)

            if K1 is None:
                raise ValueError("trigger strike is required for gap options")
            if K2 is None:
                raise ValueError("payoff strike is required for gap options")
            
            if option_type == 'call':
                trigger_valid = (np.max(paths, axis=1) >= K1)
            elif option_type == 'put':
                trigger_valid = (np.min(paths, axis=1) <= K1)

            payoffs = intrinsic_value(final_prices, K2, option_type) * trigger_valid
        
        else: 
            raise ValueError(f'Unsupported option style: {option_style}')

        price = np.exp(-r * T) * np.mean(payoffs)
        std_error = np.std(payoffs) / np.sqrt(n_paths)

        return {
            'price': price, 
            'std_error': std_error,
            'n_paths': n_paths,
            'n_steps': n_steps,
            'method': 'monte_carlo',
            'option_style': option_style
        }
import numpy as np
from ..utils.stochastic_processes.gbm import GeometricBrownianMotion
from ..utils.payoff import intrinsic_value, asian_payoff
from .base_model import PricingModel

class VanillaMonteCarlo(PricingModel):
    def calculate(self, S, K, T, r, sigma, option_type='call', q=0.0, **kwargs):
        n_paths = kwargs.get('n_paths', 10000)
        n_steps = kwargs.get('n_steps', 100)
        option_style = kwargs.get('option_style', 'european')
        seed = kwargs.get('seed', None)

        if option_style == 'asian':
            averaging_type = kwargs.get('averaging_type', 'arithmetic')
            monitoring_dates = kwargs.get('monitoring_dates', None)
            observed_values = kwargs.get('observed_values', None)
            t_today = kwargs.get('t_today', 0.0)
            
            monitored_prices = self._simulate_asian_prices(
                S, T, r, sigma, q, n_paths, n_steps, observed_values, 
                monitoring_dates, t_today, seed
            )
            payoffs = asian_payoff(monitored_prices, K, option_type, averaging_type)
            price = np.exp(-r * (T - t_today)) * np.mean(payoffs)
            std_error = np.std(payoffs) / np.sqrt(n_paths)

            return {
                'price': price, 
                'std_error': std_error,
                'n_paths': n_paths,
                'n_steps': n_steps,
                'method': 'monte_carlo',
                'option_style': option_style
            }

        paths = GeometricBrownianMotion.simulate(S, T, r, sigma, q, n_paths, n_steps, seed=seed)
        final_prices = paths[:, -1]

        if option_style == 'european':
            payoffs = intrinsic_value(final_prices, K, option_type)
        
        elif option_style == 'barrier':
            barrier_type = kwargs.get('barrier_type', None)
            barrier_level = kwargs.get('barrier_level', None)
            if barrier_type is None or barrier_level is None:
                raise ValueError("barrier_type and barrier_level are required for barrier options")

            if barrier_type == "up-and-in":
                barrier_valid = np.max(paths, axis=1) >= barrier_level
            elif barrier_type == "up-and-out":
                barrier_valid = np.max(paths, axis=1) < barrier_level
            elif barrier_type == "down-and-in":
                barrier_valid = np.min(paths, axis=1) <= barrier_level
            elif barrier_type == "down-and-out":
                barrier_valid = np.min(paths, axis=1) > barrier_level
            else:
                raise ValueError("Invalid barrier_type")
            
            payoffs = intrinsic_value(final_prices, K, option_type) * barrier_valid
        
        elif option_style == 'gap':
            K1 = kwargs.get('K1', None)
            K2 = kwargs.get('K2', None)
            if K1 is None or K2 is None:
                raise ValueError("K1 (trigger) and K2 (payoff) are required for gap options")

            if option_type == 'call':
                trigger_valid = np.max(paths, axis=1) >= K1
            elif option_type == 'put':
                trigger_valid = np.min(paths, axis=1) <= K1
            else:
                raise ValueError("option_type must be 'call' or 'put'")
            
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
    
    def _simulate_asian_prices(self, S, T, r, sigma, q, n_paths, n_steps, observed_values=None, monitoring_dates=None, t_today=0.0, seed=None):
        if monitoring_dates is None:
            monitoring_dates = np.linspace(0, T, n_steps + 1)
        elif isinstance(monitoring_dates, int):
            monitoring_dates = np.linspace(0, T, monitoring_dates)
        else:
            monitoring_dates = np.array(monitoring_dates)

        if observed_values is None:
            observed_values = []
        observed_values = np.array(observed_values, dtype=np.float64)
        n_observed = len(observed_values)

        if len(monitoring_dates[monitoring_dates <= t_today]) != n_observed:
            raise ValueError(f'valeurs observées {n_observed} vs valeurs attendues {len(monitoring_dates[monitoring_dates <= t_today])}')

        future_monitoring_dates = monitoring_dates[monitoring_dates > t_today]
        n_steps_future = int(n_steps * (T - t_today) / T)
        paths_future = GeometricBrownianMotion.simulate(S, T - t_today, r, sigma, q, n_paths, n_steps_future, seed=seed)
        time_points_future = np.linspace(0, T - t_today, n_steps_future + 1)

        monitored_future_prices = np.array([np.interp(future_monitoring_dates - t_today, time_points_future, path) for path in paths_future])

        if n_observed > 0:
            monitored_prices = np.hstack([
                np.tile(observed_values, (n_paths, 1)),
                monitored_future_prices
            ])
        else:
            monitored_prices = monitored_future_prices

        return monitored_prices

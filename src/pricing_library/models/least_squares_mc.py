import numpy as np
from ..utils.stochastic_processes.gbm import GeometricBrownianMotion
from ..utils.payoff import intrinsic_value
from ..utils.regression import get_regressor
from .base_model import PricingModel

class LeastSquaresMC(PricingModel):    
    def calculate(self, S, K, T, r, sigma, option_type='call', q=0.0, **kwargs):
        n_paths = kwargs.get('n_paths', 10000)
        n_steps = kwargs.get('n_steps', 100)
        regression_type = kwargs.get('regression_type', 'polynomial')
        regression_degree = kwargs.get('regression_degree', 2)
        seed = kwargs.get('seed', None)
        
        paths = GeometricBrownianMotion.simulate(S, T, r, sigma, q, n_paths, n_steps, seed=seed)
        
        price, std_error = self._lsm_pricing(
            paths, K, r, T, option_type, regression_type, regression_degree
        )
        
        return {
            'price': price,
            'std_error': std_error,
            'n_paths': n_paths,
            'n_steps': n_steps,
            'method': 'least_squares_mc',
            'option_style': 'american',
            'regression_type': regression_type,
            'regression_degree': regression_degree
        }
    
    def _lsm_pricing(self, paths, K, r, T, option_type, regression_type, degree):
        num_paths, num_steps = paths.shape
        dt = T / (num_steps - 1)
        discount_factor = np.exp(-r * dt)
      
        regressor = get_regressor(regression_type, degree=degree)
        
        exercise_values = intrinsic_value(paths, K, option_type)
        
        option_values = np.zeros_like(paths)
        option_values[:, -1] = exercise_values[:, -1]  
        
        for t in range(num_steps - 2, -1, -1):
            discounted_next_values = option_values[:, t + 1] * discount_factor
            
            itm_mask = exercise_values[:, t] > 0
            
            if np.any(itm_mask):
                itm_paths = paths[itm_mask, t]
                itm_discounted_values = discounted_next_values[itm_mask]

                continuation_estimates = regressor.fit_predict(
                    itm_paths, itm_discounted_values, itm_paths
                )
                exercise_decision = exercise_values[itm_mask, t] > continuation_estimates
                
                option_values[itm_mask, t] = np.where(
                    exercise_decision,
                    exercise_values[itm_mask, t],  
                    discounted_next_values[itm_mask]  
                )
            
            otm_mask = ~itm_mask
            option_values[otm_mask, t] = discounted_next_values[otm_mask]
        
        price = np.mean(option_values[:, 0])
        payoffs = option_values[:, 0]
        std_error = np.std(payoffs) / np.sqrt(num_paths)
        
        return price, std_error
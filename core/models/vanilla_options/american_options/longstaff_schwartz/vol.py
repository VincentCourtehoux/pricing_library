import numpy as np
from scipy.optimize import brentq, minimize_scalar
import time
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..')))
from core.models.vanilla_options.american_options.longstaff_schwartz.pricing import lsm_us_premium

class AmericanImpliedVolSolver:
    
    def __init__(self, S0, K, T, r, q, option_type, market_price,
                 N=50, nb_paths=50000, degree=2, seed=42):
        """
        Initialize the solver with option parameters and market price.

        Parameters:
        S0 (float): Spot price of the underlying asset.
        K (float): Strike price.
        T (float): Time to maturity (in years).
        r (float): Risk-free interest rate.
        q (float): Continuous dividend yield.
        option_type (str): 'call' or 'put'.
        market_price (float): Observed market price of the option.
        N (int): Number of time steps for the Monte Carlo simulation.
        nb_paths (int): Number of Monte Carlo simulation paths.
        degree (int): Degree of Laguerre polynomials for regression.
        seed (int): Random seed for reproducibility.
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.q = q
        self.option_type = option_type
        self.market_price = market_price
        self.N = N
        self.nb_paths = nb_paths
        self.degree = degree
        self.seed = seed
        
        self.n_evaluations = 0
        self.evaluation_times = []
        
    def price_function(self, sigma):
        """
        Compute the difference between theoretical option price (via LSM) and market price.

        Parameters:
        sigma (float): Volatility to evaluate.

        Returns:
        float: Theoretical price minus market price.
        """
        start_time = time.time()
        
        try:
            theoretical_price = lsm_us_premium(
                self.S0, self.K, self.r, sigma, self.T, self.q,
                self.N, self.nb_paths, self.option_type, self.degree, self.seed
            )
            
            self.n_evaluations += 1
            self.evaluation_times.append(time.time() - start_time)
            
            return theoretical_price - self.market_price
            
        except Exception as e:
            return float('inf')
    
    def objective_function(self, sigma):
        """
        Objective function for minimization: squared difference between theoretical and market price.

        Parameters:
        sigma (float): Volatility to evaluate.

        Returns:
        float: Squared error.
        """
        return self.price_function(sigma)**2
    
    def solve_brentq(self, vol_min=0.01, vol_max=3.0, tolerance=1e-6, max_iter=100):
        """
        Find implied volatility by root-finding using Brent’s method.

        Parameters:
        vol_min (float): Minimum volatility bound.
        vol_max (float): Maximum volatility bound.
        tolerance (float): Convergence tolerance.
        max_iter (int): Maximum iterations allowed.

        Returns:
        dict: Result including implied volatility, success flag, number of evaluations, and timing.
        """        
        try:
            f_min = self.price_function(vol_min)
            f_max = self.price_function(vol_max)
            
            if f_min * f_max > 0:
                raise ValueError(f"No sign change detected in interval [{vol_min}, {vol_max}]")

            start_time = time.time()
            implied_vol = brentq(
                self.price_function, vol_min, vol_max,
                xtol=tolerance, maxiter=max_iter
            )
            solve_time = time.time() - start_time
            
            return {
                'implied_vol': implied_vol,
                'method': 'Brent',
                'success': True,
                'n_evaluations': self.n_evaluations,
                'solve_time': solve_time,
                'final_error': abs(self.price_function(implied_vol)),
                'avg_evaluation_time': np.mean(self.evaluation_times)
            }
            
        except Exception as e:
            return {
                'implied_vol': None,
                'method': 'Brent',
                'success': False,
                'error': str(e),
                'n_evaluations': self.n_evaluations
            }
    
    def solve_minimize(self, vol_bounds=(0.01, 3.0), tolerance=1e-6):
        """
        Find implied volatility by minimizing the squared error function using bounded scalar minimization.

        Parameters:
        vol_bounds (tuple): Bounds for volatility search.
        tolerance (float): Convergence tolerance.

        Returns:
        dict: Result including implied volatility, success flag, number of evaluations, and timing.
        """        
        try:
            start_time = time.time()
            
            result = minimize_scalar(
                self.objective_function,
                bounds=vol_bounds,
                method='bounded',
                options={'xatol': tolerance}
            )
            
            solve_time = time.time() - start_time
            
            return {
                'implied_vol': result.x,
                'method': 'Minimize',
                'success': result.success,
                'n_evaluations': self.n_evaluations,
                'solve_time': solve_time,
                'final_error': np.sqrt(result.fun),
                'avg_evaluation_time': np.mean(self.evaluation_times),
                'scipy_result': result
            }
            
        except Exception as e:
            return {
                'implied_vol': None,
                'method': 'Minimize',
                'success': False,
                'error': str(e),
                'n_evaluations': self.n_evaluations
            }
    
    def solve_adaptive(self, initial_bounds=(0.01, 3.0), tolerance=1e-6):
        """
        Adaptive solver trying Brent’s method first, then falling back to minimization if Brent fails.

        Parameters:
        initial_bounds (tuple): Initial volatility bounds for the search.
        tolerance (float): Convergence tolerance.

        Returns:
        dict: Result including implied volatility, success flag, number of evaluations, and timing.
        """
        self.n_evaluations = 0
        self.evaluation_times = []

        result_brent = self.solve_brentq(
            initial_bounds[0], initial_bounds[1], tolerance
        )
        
        if result_brent['success']:
            return result_brent

        print("Brent method failed, trying minimization...")
        self.n_evaluations = 0
        self.evaluation_times = []

        result_minimize = self.solve_minimize(initial_bounds, tolerance)
        
        if result_minimize['success']:
            return result_minimize
        else:
            print("All methods failed")
            return result_minimize
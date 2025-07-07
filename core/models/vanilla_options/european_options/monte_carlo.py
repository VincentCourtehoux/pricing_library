import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from models.utils.gbm_simulation import simulate_gbm
from models.utils.payoff import compute_payoff
from models.utils.confidence_interval import confidence_interval
from models.vanilla_options.european_options.black_scholes.pricing_scalar import BlackScholesScalar
bs = BlackScholesScalar()

def mc_european_premium(S0, K, T, r, sigma, q, N, nb_paths, option_type="call", return_all=False, seed=None):
    """
    Estimates the price of a European option using Monte Carlo simulation.

    Parameters:
    S0 (float): Initial asset price
    K (float): Strike price
    T (float): Time to maturity (in years)
    r (float): Risk-free interest rate
    sigma (float): Volatility of the asset
    q (float): Continuous dividend yield
    N (int): Number of time steps
    nb_paths (int): Number of simulated paths
    option_type (str): 'call' or 'put'
    return_all (bool): If True, returns detailed results

    Returns:
    float: Option price estimate (if return_all=False)
    dict: Contains price, confidence interval, paths, payoffs, and discounted payoffs (if return_all=True)
    """
    paths = simulate_gbm(S0, T, r, sigma, q, N, nb_paths, seed)
    
    ST = paths[:, -1]
    payoffs = compute_payoff(ST, K, option_type)
    discounted_payoffs = np.exp(-r * T) * payoffs
    mean_price, margin = confidence_interval(discounted_payoffs)
    bs_price = bs.bs_european_scalar_premium(S0, K, T, r, sigma, q, option_type)
    
    if return_all:
        return {
            "price": mean_price,
            "confidence_interval": (mean_price - margin, mean_price + margin),
            "margin_error": margin,
            "black_scholes_price": bs_price,
            "error_vs_bs": abs(mean_price - bs_price),
            "relative_error": abs(mean_price - bs_price) / bs_price * 100 if bs_price != 0 else float('inf'),
            "paths": paths,
            "final_prices": ST,
            "payoffs": payoffs,
            "discounted_payoffs": discounted_payoffs,
            "statistics": {
                "mean_ST": np.mean(ST),
                "std_ST": np.std(ST),
                "mean_payoff": np.mean(payoffs),
                "std_payoff": np.std(payoffs),
                "nb_itm": np.sum(payoffs > 0)
            }
        }
    else:
        return mean_price
    
def convergence_analysis(S0, K, T, r, sigma, q, option_type="call", 
                        path_counts=None, N=100):
    """
    Perform convergence analysis of a Monte Carlo European option pricing.

    Parameters:
    S0 (float): Initial asset price.
    K (float): Strike price.
    T (float): Time to maturity (in years).
    r (float): Risk-free interest rate.
    sigma (float): Volatility of the underlying asset.
    q (float): Continuous dividend yield.
    option_type (str, optional): Type of option, 'call' or 'put'. Default is 'call'.
    path_counts (list of int, optional): List of numbers of Monte Carlo simulation paths to use. Default is None.
    N (int, optional): Number of time steps in the Monte Carlo simulation. Default is 100.

    Returns:
    results (list of dict): Each dictionary contains Monte Carlo price, Black-Scholes price, absolute error, and relative error for each number of paths.
    bs_price (float): Black-Scholes analytical price of the European option.
    """
    if path_counts is None:
        path_counts = [1000, 5000, 10000, 25000, 50000, 100000]
    
    bs_price = bs.bs_european_scalar_premium(S0, K, T, r, sigma, q, option_type)
    
    results = []
    for nb_paths in path_counts:
        mc_price = mc_european_premium(
            S0, K, T, r, sigma, q, N, nb_paths, 
            option_type, return_all=False, seed=None
        )
        
        error = abs(mc_price - bs_price)
        relative_error = error / bs_price * 100 if bs_price != 0 else float('inf')
        
        results.append({
            'nb_paths': nb_paths,
            'mc_price': mc_price,
            'bs_price': bs_price,
            'error': error,
            'relative_error': relative_error
        })
    
    return results, bs_price
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from models.utils.gbm_simulation import simulate_gbm
from models.utils.payoff import compute_payoff
from models.utils.laguerre_matrix import laguerre_matrix


def longstaff_schwartz_american(S0, K, r, sigma, T, q, N, nb_paths, option_type='call', degree=2, seed=None):
    """
    Prices an American option using the Longstaff-Schwartz Monte Carlo method.

    Parameters:
    - S0 (float): Initial asset price
    - K (float): Strike price
    - r (float): Risk-free interest rate
    - sigma (float): Volatility
    - T (float): Time to maturity (in years)
    - q (float): Dividend yield
    - N (int): Number of time steps
    - nb_paths (int): Number of Monte Carlo simulation paths
    - option_type (str): 'call' or 'put'
    - degree (int): Degree of Laguerre polynomials used in regression
    - seed (int or None): Random seed for reproducibility

    Returns:
    - float: Estimated price of the American option at time 0
    """
    dt = T / N
    discount = np.exp(-r * dt)
    
    S = simulate_gbm(S0, r, sigma, T, q, N, nb_paths, seed)
    
    CF = compute_payoff(S[-1], K, option_type)
    
    for t in range(N-1, 0, -1):
        CF = CF * discount
        
        exercise_value = compute_payoff(S[t], K, option_type)
        itm = exercise_value > 0
        
        if np.sum(itm) == 0:
            continue
            
        S_itm = S[t, itm]
        CF_itm = CF[itm]
        exercise_value_itm = exercise_value[itm]
        
        if len(S_itm) > degree + 1: 
            X = laguerre_matrix(S_itm, degree)
            try:
                coeffs = np.linalg.lstsq(X, CF_itm, rcond=None)[0]
                continuation_value = X @ coeffs
            except np.linalg.LinAlgError:
                continuation_value = CF_itm
        else:
            continuation_value = np.full(len(S_itm), np.mean(CF_itm))
        
        exercise = exercise_value_itm > continuation_value
        
        CF[itm] = np.where(exercise, exercise_value_itm, CF_itm)
    
    CF = CF * discount
    
    price = np.mean(CF)
    return price
    
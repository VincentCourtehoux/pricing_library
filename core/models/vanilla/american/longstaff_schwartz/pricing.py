import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
from models.utils.gbm_simulation import simulate_gbm
from models.utils.payoff import compute_payoff
from models.utils.laguerre_matrix import laguerre_matrix

def longstaff_schwartz_american(S0, K, r, sigma, T, q, N, nb_paths, option_type='call', degree=2, seed=None):
    """
    Prices an American option using the Longstaff-Schwartz Monte Carlo method.
    Fixed version that follows the original LS algorithm more closely.
    
    Parameters:
    S0 (float): Initial asset price
    K (float): Strike price
    r (float): Risk-free interest rate
    sigma (float): Volatility
    T (float): Time to maturity (in years)
    q (float): Dividend yield
    N (int): Number of time steps
    nb_paths (int): Number of Monte Carlo simulation paths
    option_type (str): 'call' or 'put'
    degree (int): Degree of Laguerre polynomials used in regression
    seed (int or None): Random seed for reproducibility
    
    Returns:
    dict: Contains 'price', 'std_error', 'european_price', and diagnostics
    """
    if seed is not None:
        np.random.seed(seed)
    
    dt = T / N
    discount = np.exp(-r * dt)
    S = simulate_gbm(S0, T, r, sigma, q, N, nb_paths, seed)
    CF = compute_payoff(S[:, -1], K, option_type) 
    
    for t in range(N - 1, 0, -1):
        CF = CF * discount
        exercise_value = compute_payoff(S[:, t], K, option_type)
        itm = exercise_value > 0

        if np.sum(itm) == 0:
            continue

        S_itm = S[itm, t]
        CF_itm = CF[itm]  
        exercise_value_itm = exercise_value[itm]
        
        if len(S_itm) > degree + 1:
            try:
                X = laguerre_matrix(S_itm, degree)

                if X.shape[0] >= X.shape[1]:
                    XTX = X.T @ X
                    XTy = X.T @ CF_itm
                    reg = 1e-12 * np.trace(XTX) / XTX.shape[0]
        
                    try:
                        coeffs = np.linalg.solve(XTX + reg * np.eye(XTX.shape[0]), XTy)
                    except np.linalg.LinAlgError:
                        coeffs = np.linalg.pinv(X) @ CF_itm
                    continuation_value = X @ coeffs
                else:
                    continuation_value = np.full(len(S_itm), np.mean(CF_itm))
            except (np.linalg.LinAlgError, ValueError):
                continuation_value = np.full(len(S_itm), np.mean(CF_itm))
        else:
            continuation_value = np.full(len(S_itm), np.mean(CF_itm))
    
        exercise_decision = exercise_value_itm >= continuation_value
        CF[itm] = np.where(exercise_decision, exercise_value_itm, CF_itm)
    CF = CF * discount
    price = np.mean(CF)
    
    return price

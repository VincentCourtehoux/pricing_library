import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.gbm_simulation import simulate_gbm

def mc_asian_premium(S0, K, T, r, sigma, q, N, nb_paths, option_type, seed=None):
    """
    Prices an Asian option (average price) using simulated GBM paths.

    Parameters:
    S0: Initial asset price
    K: Strike price
    T: Maturity in years
    r: Risk-free rate
    sigma: Volatility
    q: Dividend yield
    N: Number of time steps
    nb_paths: Number of simulated paths
    option_type: "call" for Call, "put" for Put

    Returns:
    float: Present value of the Asian option
    """
    S_paths = simulate_gbm(S0, T, r, sigma, q, N, nb_paths, seed)
    
    if option_type == "call":
        payoffs = np.maximum(np.mean(S_paths, axis=1) - K, 0)
    elif option_type == "put":
        payoffs = np.maximum(K - np.mean(S_paths, axis=1), 0)
    else:
        raise ValueError("Invalid option type")

    return np.exp(-r * T) * np.mean(payoffs)
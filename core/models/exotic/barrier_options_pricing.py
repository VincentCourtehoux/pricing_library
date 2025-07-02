import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.payoff import compute_payoff
from utils.gbm_simulation import simulate_gbm

def is_barrier_valid(paths, barrier, barrier_type):
    """
    Check whether each simulated price path satisfies the barrier condition.

    Parameters:
    paths (np.ndarray): Array of simulated price paths, shape (nb_paths, N+1)
    barrier (float): Barrier level price
    barrier_type (str): Barrier option type, one of "up-in", "up-out", "down-in", "down-out"

    Returns:
    np.ndarray: Boolean array of shape (nb_paths,), True if barrier condition satisfied per path
    """
    if barrier_type == "up-in":
        return np.max(paths, axis=1) >= barrier
    elif barrier_type == "up-out":
        return np.max(paths, axis=1) < barrier
    elif barrier_type == "down-in":
        return np.min(paths, axis=1) <= barrier
    elif barrier_type == "down-out":
        return np.min(paths, axis=1) > barrier
    else:
        raise ValueError("Invalid barrier_type")
      
def barrier_option_premium(S0, K, T, r, sigma, q, barrier, N, nb_paths, option_type, barrier_type, seed=None):
    """
    Compute the price of a barrier option using Monte Carlo simulation.

    Parameters:
    S0 (float): Initial underlying asset price
    K (float): Strike price
    T (float): Time to maturity in years
    r (float): Risk-free interest rate (annual)
    sigma (float): Volatility of the underlying (annual)
    q (float): Continuous dividend yield (annual)
    barrier (float): Barrier level price
    N (int): Number of time steps
    nb_paths (int): Number of Monte Carlo simulation paths
    option_type (str): Option type, "call" or "put"
    barrier_type (str): Barrier type, one of "up-in", "up-out", "down-in", "down-out"
    seed (int or None): Random seed for reproducibility

    Returns:
    float: Estimated price (premium) of the barrier option
    """
    if seed is not None:
        np.random.seed(seed)
    paths = simulate_gbm(S0, T, r, sigma, q, N, nb_paths)
    payoffs = compute_payoff(paths[:, -1], K, option_type)
    valid = is_barrier_valid(paths, barrier, barrier_type)
    final_payoffs = payoffs * valid 
    discounted_payoffs = final_payoffs * np.exp(-r * T)
    price = np.mean(discounted_payoffs)
    return price


print(barrier_option_premium(100, 100, 1, 0.05, 0.2, 0, 200, 252, 100000, 'call', 'down-out'))

import numpy as np

def simulate_gbm(S0, T, r, sigma, q, N, nb_paths, seed=None):
    """
    Simulates asset price paths using the geometric Brownian motion model.

    Parameters:
    S0 (float): Initial asset price
    T (float): Time to maturity (in years)
    r (float): Risk-free interest rate
    sigma (float): Volatility of the asset
    q (float): Continuous dividend yield
    N (int): Number of time steps
    nb_paths (int): Number of simulated paths

    Returns:
    np.ndarray: Simulated paths of shape (nb_sim, N+1)
    """
    if seed is not None:
        np.random.seed(seed)
    
    dt = T / N

    S = np.zeros((nb_paths, N + 1))
    S[:, 0] = S0

    Z = np.random.standard_normal((nb_paths, N))
    
    for t in range(N):
        S[:, t+1] = S[:, t] * np.exp(
            (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t]
        )
    
    return S
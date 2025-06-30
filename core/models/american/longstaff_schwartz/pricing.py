import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from models.utils.path_simulation import simulate_gbm_paths

def lsm(S0, K, T, r, sigma, q=0.0, option_type="call", N=252, nb_paths=10000):
    """
    Price an American option using the Longstaff-Schwartz method.

    Parameters:
    S0 (float): initial asset price
    K (float): strike price
    T (float): time to maturity (years)
    r (float): risk-free interest rate
    sigma (float): volatility
    q (float): continuous dividend yield (default 0.0)
    option_type (str): "call" or "put" (default "call")
    N (int): number of time steps (default 252)
    nb_paths (int): number of Monte Carlo paths (default 10000)

    Returns:
    float: estimated option price
    """
    dt = T / N
    discount = np.exp(-r * dt)

    paths = simulate_gbm_paths(S0, T, r, sigma, q, N, nb_paths)

    if option_type == "call":
        payoff = np.maximum(paths[:, -1] - K, 0)
    else:
        payoff = np.maximum(K - paths[:, -1], 0)

    V = payoff.copy()

    for t in reversed(range(N)):
        St = paths[:, t]

        if option_type == "call":
            mask_itm = St > K
            immediate_exercise = np.maximum(St - K, 0)
        else:
            mask_itm = St < K
            immediate_exercise = np.maximum(K - St, 0)

        V = V * discount

        if not np.any(mask_itm):
            continue

        Y = V[mask_itm]
        X = np.vstack([np.ones(np.sum(mask_itm)), St[mask_itm], St[mask_itm]**2]).T

        beta, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None)
        continuation_value = X.dot(beta)

        exercise_now = immediate_exercise[mask_itm] > continuation_value

        V[mask_itm] = np.where(exercise_now, immediate_exercise[mask_itm], V[mask_itm])

    price = np.mean(V)
    return price
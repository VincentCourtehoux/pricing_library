import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from core.models.utils.path_simulation import simulate_gbm_paths
from core.models.utils.payoff import compute_payoff
from core.models.utils.statistics import confidence_interval

def price_european_option_mc(S0, K, T, r, sigma, q, N, nb_paths, option_type="call", return_all=False):
    paths = simulate_gbm_paths(S0, T, r, sigma, q, N, nb_paths)
    ST = paths[:, -1]
    payoff = compute_payoff(ST, K, option_type)
    discounted = np.exp(-r * T) * payoff
    mean, margin = confidence_interval(discounted)

    if return_all:
        return {
            "price": mean,
            "confidence_interval": (mean - margin, mean + margin),
            "paths": paths,
            "payoffs": payoff,
            "discounted_payoffs": discounted
        }
    else:
        return mean
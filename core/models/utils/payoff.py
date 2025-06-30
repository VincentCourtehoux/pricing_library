import numpy as np

def compute_payoff(ST, K, option_type="call"):
    if option_type == "call":
        return np.maximum(ST - K, 0)
    elif option_type == "put":
        return np.maximum(K - ST, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
import numpy as np

def compute_payoff(S, K, option_type):
    """
    Compute the payoff of a European option.

    Parameters:
    S (array-like or float): Asset price(s) at maturity.
    K (float): Strike price of the option.
    option_type (str): Type of the option, either 'call' or 'put' (case-insensitive).

    Returns:
    payoff (array-like or float): The payoff value(s) of the option.
    """
    if option_type.lower() == 'call':
        return np.maximum(S - K, 0)
    elif option_type.lower() == 'put':
        return np.maximum(K - S, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
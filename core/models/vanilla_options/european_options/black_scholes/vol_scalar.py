import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..')))
from core.models.vanilla_options.european_options.black_scholes.pricing_scalar import BlackScholesScalar

bs = BlackScholesScalar()

def implied_volatility_scalar(market_price, S, K, T, r, q=0.0, option_type="call", tol=1e-6, max_iterations=100, sigma_initial=0.2):
    """
    Compute the implied volatility for a single European option using the Newton-Raphson method.

    Parameters:
    market_price (float): Observed market price of the option
    S (float): Current price of the underlying asset
    K (float): Strike price
    T (float): Time to maturity (in years)
    r (float): Risk-free interest rate (annualized)
    q (float): Continuous dividend yield
    option_type (str): Option type, either 'call' or 'put'
    tol (float): Convergence tolerance
    max_iterations (int): Maximum iterations allowed
    sigma_initial (float): Starting guess for volatility

    Returns:
    float: Implied volatility estimate (or np.nan if failed)
    """
    if T == 0:
        intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
        return 0.0 if abs(market_price - intrinsic) < 1e-6 else np.nan

    sigma = sigma_initial

    for _ in range(max_iterations):
        price = bs.bs_eu_scalar_premium(S, K, T, r, sigma, q, option_type)
        vega_val = bs.vega(S, K, T, r, sigma, q)
        vega_val = max(vega_val, 1e-8)  
        
        price_diff = price - market_price
        sigma_new = sigma - price_diff / vega_val
        sigma_new = np.clip(sigma_new, 1e-6, 5.0)

        if abs(sigma_new - sigma) < tol:
            if sigma_new >= 5 - 1e-5 or sigma_new <= 1e-6 + 1e-5:
                return np.nan  
            return sigma_new

        sigma = sigma_new

    return np.nan 

import numpy as np
from core.models.european.black_scholes.pricing_vectorized import BlackScholesVectorized

bs = BlackScholesVectorized()

def implied_volatility_vectorized(market_prices, S, K, T, r, q, option_types, tolerance=1e-4, max_iterations=1000, sigma_initial=0.2):
    """
    Compute the implied volatilities for multiple options using the Newton-Raphson method.

    Parameters:
    market_prices (np.ndarray): Observed market prices of the options.
    S (np.ndarray): Current prices of the underlying assets.
    K (np.ndarray): Strike prices of the options.
    T (np.ndarray): Times to maturity (in years).
    r (np.ndarray): Risk-free interest rates (annualized).
    q (np.ndarray): Continuous dividend yields
    option_types (np.ndarray of str): Option types, each 'call' or 'put'.
    tolerance (float): Convergence tolerance for volatility changes.
    max_iterations (int): Maximum number of Newton-Raphson iterations.
    sigma_initial (float): Initial guess for the volatilities.

    Returns:
    np.ndarray: Estimated implied volatilities for each option.
    """
    market_prices = np.asarray(market_prices, dtype=float)
    S             = np.asarray(S, dtype=float)
    K             = np.asarray(K, dtype=float)
    T             = np.asarray(T, dtype=float)
    r             = np.asarray(r, dtype=float)
    q             = np.asarray(q, dtype=float)
    option_types  = np.asarray(option_types)

    sigma = np.full_like(market_prices, sigma_initial, dtype=float)
    valid_mask = T > 0
    expired_mask = ~valid_mask

    # Handle expired options immediately
    intrinsic_call = np.maximum(S - K, 0)
    intrinsic_put = np.maximum(K - S, 0)
    intrinsic_value = np.where(option_types == "call", intrinsic_call, intrinsic_put)

    sigma[expired_mask] = np.where(
        np.isclose(market_prices[expired_mask], intrinsic_value[expired_mask], atol=1e-6),
        0.0,  # exact match → 0 vol
        np.nan  # mismatch → invalid input
    )

    # Newton-Raphson for valid (non-expired) options
    for _ in range(max_iterations):
        if not np.any(valid_mask):  # all expired? skip loop
            break

        price_estimates = bs.premium(S[valid_mask], K[valid_mask], T[valid_mask], r[valid_mask], sigma[valid_mask], q[valid_mask], option_types[valid_mask])
        vegas = bs.vega(S[valid_mask], K[valid_mask], T[valid_mask], r[valid_mask], sigma[valid_mask], q[valid_mask])
        price_differences = price_estimates - market_prices[valid_mask]

        vegas = np.where(vegas < 1e-8, 1e-8, vegas)
        sigma_update = sigma[valid_mask] - price_differences / vegas
        sigma_update = np.clip(sigma_update, 1e-6, 5.0)

        if np.all(np.abs(sigma_update - sigma[valid_mask]) < tolerance):
            sigma[valid_mask] = sigma_update
            break

        sigma[valid_mask] = sigma_update
    else:
        raise RuntimeError("Implied volatility did not converge after max_iterations.")

    # Flag out-of-bounds volatilities as NaN
    out_of_bounds = (sigma >= 5.0 - 1e-5) | (sigma <= 1e-6 + 1e-5)
    sigma[out_of_bounds] = np.nan

    return sigma
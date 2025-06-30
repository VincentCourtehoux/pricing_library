from scipy.optimize import brentq

from pricing import lsm

def vega_numerical(S, K, T, r, sigma, q=0.0, option_type="call", N=252, nb_paths=100, h=0.1):
    """
    Compute the numerical Vega of an option priced by Longstaff-Schwartz.

    Parameters:
    S (float): initial underlying asset price
    K (float): strike price
    T (float): time to maturity (in years)
    r (float): risk-free interest rate
    sigma (float): volatility used for pricing
    q (float): continuous dividend yield (default 0.0)
    option_type (str): "call" or "put" (default "call")
    N (int): number of time steps (default 252)
    nb_paths (int): number of Monte Carlo paths (default 100)
    h (float): finite difference step size for numerical derivative (default 0.1)

    Returns:
    float: numerical estimate of Vega (sensitivity of option price to volatility)
    """
    price_up = lsm(S, K, T, r, sigma + h, q, N, nb_paths, option_type)
    price_down = lsm(S, K, T, r, sigma - h, q, N, nb_paths, option_type)
    return (price_up - price_down) / (2 * h)

def implied_vol_lsm(market_price, S0, K, T, r, q=0.0, option_type="call", N=252, nb_paths=100, vol_bounds=(1e-6, 3.0), tol=1e-4, max_iter=20):
    """
    Calculate the implied volatility of an American option using Longstaff-Schwartz.

    Parameters:
    market_price (float): observed market price
    S0 (float): initial underlying asset price
    K (float): strike price
    T (float): maturity (in years)
    r (float): risk-free interest rate
    q (float): continuous dividend yield
    option_type (str): "call" or "put"
    N (int): number of time steps (default 252)
    nb_paths (int): number of Monte Carlo paths (default 100)
    vol_bounds (tuple): bounds for the volatility search
    tol (float): tolerance for stopping criterion on price
    max_iter (int): maximum number of Newton iterations

    Returns:
    float: estimated implied volatility
    """
    def objective(sigma):
        return lsm(S0, K, T, r, sigma, q, N, nb_paths, option_type) - market_price

    sigma_est = brentq(objective, *vol_bounds, xtol=tol)

    for i in range(max_iter):
        price = lsm(S0, K, T, r, sigma_est, q, N, nb_paths, option_type)
        diff = price - market_price
        if abs(diff) < tol:
            break
        vega = vega_numerical(S0, K, T, r, sigma_est, q, option_type, N, nb_paths)
        if vega == 0:
            break
        sigma_est -= diff / vega

        sigma_est = max(vol_bounds[0], min(vol_bounds[1], sigma_est))
    return sigma_est

print(implied_vol_lsm(4.4765, 36, 40, 1, 0.06, 0, "put", 252, 10000))
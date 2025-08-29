import numpy as np
from .black_scholes import BlackScholesModel
from scipy.stats import norm

class ImpliedVolatility:

    @staticmethod
    def calculate(price, S, K, T, r, option_type='call', q=0.0, max_iterations=100, tol=1e-6, vol_bounds=(1e-6, 5.0)):
        
        bs_model = BlackScholesModel()
        vol_min, vol_max = vol_bounds
        guess = 0.2  
        lower, upper = vol_min, vol_max

        for i in range(max_iterations):
            bs_price = bs_model.calculate(S, K, T, r, guess, option_type, q)['price']
            price_diff = price - bs_price

            d1 = (np.log(S / K) + (r - q + 0.5 * guess**2) * T) / (guess * np.sqrt(T))
            vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

            if vega > 1e-8:
                guess += price_diff / vega
            else:
                guess = (lower + upper) / 2

            guess = max(vol_min, min(vol_max, guess))

            if bs_price < price:
                lower = guess
            else:
                upper = guess

            if abs(price_diff) < tol:
                return guess

        return guess


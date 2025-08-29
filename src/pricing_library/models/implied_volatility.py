import numpy as np
from .black_scholes import BlackScholesModel
from scipy.stats import norm

class ImpliedVolatility:

    @staticmethod
    def calculate(params, max_iterations=100, tol=1e-6, vol_bounds=(1e-6, 5.0)):
        S = params['S']
        K = params['K']
        T = params['T']
        r = params['r']
        market_price = params['price']
        option_type = params.get('option_type', 'call')
        q = params.get('q', 0.0)

        bs_model = BlackScholesModel()
        vol_min, vol_max = vol_bounds
        guess = 0.2
        lower, upper = vol_min, vol_max

        for i in range(max_iterations):
            bs_price = bs_model.calculate(**{
                'S': S,
                'K': K,
                'T': T,
                'r': r,
                'sigma': guess,
                'option_type': option_type,
                'q': q
            })['price']

            price_diff = market_price - bs_price
            d1 = (np.log(S / K) + (r - q + 0.5 * guess**2) * T) / (guess * np.sqrt(T))
            vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

            if vega > 1e-8:
                guess += price_diff / vega
            else:
                guess = (lower + upper) / 2

            guess = max(vol_min, min(vol_max, guess))

            if bs_price < market_price:
                lower = guess
            else:
                upper = guess

            if abs(price_diff) < tol:
                return guess

        return guess

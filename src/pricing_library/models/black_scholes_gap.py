import numpy as np
from scipy.stats import norm
from .base_model import PricingModel
from ..utils import gap_payoff

class BlackScholesGapModel(PricingModel):
    
    def calculate(self, S, K, T, r, sigma, option_type='call', q=0.0, **kwargs):
        K1 = kwargs.get('K1', K)
        K2 = kwargs.get('K2', K)

        S = np.atleast_1d(S).astype(float)
        K1 = np.atleast_1d(K1).astype(float)
        K2 = np.atleast_1d(K2).astype(float)
        T = np.atleast_1d(T).astype(float)
        r = np.atleast_1d(r).astype(float)
        sigma = np.atleast_1d(sigma).astype(float)
        q = np.atleast_1d(q).astype(float)
        option_type = np.atleast_1d(option_type)

        price = np.zeros_like(S, dtype=float)
        expired = T <= 0
        alive = ~expired

        if np.any(expired):
            price[expired] = gap_payoff(S[expired], K1[expired], K2[expired], option_type[expired])

        if np.any(alive):
            d1 = (np.log(S[alive] / K1[alive]) +
                  (r[alive] - q[alive] + 0.5 * sigma[alive] ** 2) * T[alive]) / (sigma[alive] * np.sqrt(T[alive]))
            d2 = d1 - sigma[alive] * np.sqrt(T[alive])

            call_price = S[alive] * np.exp(-q[alive] * T[alive]) * norm.cdf(d1) - K2[alive] * np.exp(-r[alive] * T[alive]) * norm.cdf(d2)
            put_price = K2[alive] * np.exp(-r[alive] * T[alive]) * norm.cdf(-d2) - S[alive] * np.exp(-q[alive] * T[alive]) * norm.cdf(-d1)
            price[alive] = np.where(option_type[alive] == 'call', call_price, put_price)
    
        return {
            'price': price,
            'method': 'black_scholes',
            'option_style': 'gap'
        }
    
    def calculate_greeks(self, S, K, T, r, sigma, option_type='call', q=0.0, **kwargs):
        K1 = kwargs.get('K1', K)
        K2 = kwargs.get('K2', K)

        S = np.atleast_1d(S).astype(float)
        K1 = np.atleast_1d(K1).astype(float)
        K2 = np.atleast_1d(K2).astype(float)
        T = np.atleast_1d(T).astype(float)
        r = np.atleast_1d(r).astype(float)
        sigma = np.atleast_1d(sigma).astype(float)
        q = np.atleast_1d(q).astype(float)
        option_type = np.atleast_1d(option_type)

        delta = np.zeros_like(S)
        gamma = np.zeros_like(S)
        vega = np.zeros_like(S)
        theta = np.zeros_like(S)
        rho = np.zeros_like(S)

        expired = T <= 0
        alive = ~expired

        price = self.calculate(S, K, T, r, sigma, option_type, q=q, K1=K1, K2=K2)['price']

        if np.any(alive):
            d1 = (np.log(S[alive] / K1[alive]) +
                  (r[alive] - q[alive] + 0.5 * sigma[alive] ** 2) * T[alive]) / (sigma[alive] * np.sqrt(T[alive]))
            d2 = d1 - sigma[alive] * np.sqrt(T[alive])

            delta[alive] = np.where(option_type[alive] == 'call',
                                     np.exp(-q[alive] * T[alive]) * norm.cdf(d1),
                                     np.exp(-q[alive] * T[alive]) * (norm.cdf(d1) - 1))
            
            gamma[alive] = np.exp(-q[alive] * T[alive]) * norm.pdf(d1) / (S[alive] * sigma[alive] * np.sqrt(T[alive]))
            vega[alive] = S[alive] * np.exp(-q[alive] * T[alive]) * norm.pdf(d1) * np.sqrt(T[alive]) / 100

            theta[alive] = np.where(option_type[alive] == 'call',
                                     (-S[alive] * sigma[alive] * np.exp(-q[alive] * T[alive]) * norm.pdf(d1) / (2 * np.sqrt(T[alive]))
                                      - r[alive] * K2[alive] * np.exp(-r[alive] * T[alive]) * norm.cdf(d2)
                                      + q[alive] * S[alive] * np.exp(-q[alive] * T[alive]) * norm.cdf(d1)) / 365,
                                     (-S[alive] * sigma[alive] * np.exp(-q[alive] * T[alive]) * norm.pdf(d1) / (2 * np.sqrt(T[alive]))
                                      + r[alive] * K2[alive] * np.exp(-r[alive] * T[alive]) * norm.cdf(-d2)
                                      - q[alive] * S[alive] * np.exp(-q[alive] * T[alive]) * norm.cdf(-d1)) / 365)
            
            rho[alive] = np.where(option_type[alive] == 'call',
                                   K2[alive] * T[alive] * np.exp(-r[alive] * T[alive]) * norm.cdf(d2) / 100,
                                   -K2[alive] * T[alive] * np.exp(-r[alive] * T[alive]) * norm.cdf(-d2) / 100)

        return {
            'price': price,
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho
        }

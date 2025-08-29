import numpy as np
from scipy.stats import norm
from .base_model import PricingModel
from ..utils import gap_payoff

class BlackScholesGapModel(PricingModel):
    
    def calculate(self, S, K, T, r, sigma, option_type='call', q=0.0, **kwargs):
        K1 = kwargs.get('K1', K)
        K2 = kwargs.get('K2', K)

        if T <= 0:
            price = gap_payoff(S, K1, K2, option_type)
        else:
            d1 = (np.log(S / K1) + (r - q + 0.5 * sigma ** 2)) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)

            if option_type == 'call':
                price = S * np.exp(-q * T) * norm.cdf(d1) - K2 * np.exp(-r * T) * norm.cdf(d2)
            else:
                price = K2 * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

        return {
            'price': price,
            'method': 'black_scholes',
            'option_style': 'gap'
        }
    
    def calculate_greeks(self, S, K, T, r, sigma, option_type='call', q=0.0, **kwargs):
        K1 = kwargs.get('K1', K)
        K2 = kwargs.get('K2', K)
        
        d1 = (np.log(S / K1) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        price = self.calculate(S, K, T, r, sigma, option_type, **kwargs)['price']
        
        if option_type == 'call':
            delta = np.exp(-q * T) * norm.cdf(d1)
            gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
            vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100 
            theta = (-S * sigma * np.exp(-q * T) * norm.pdf(d1) / (2 * np.sqrt(T))
                    - r * K2 * np.exp(-r * T) * norm.cdf(d2)
                    + q * S * np.exp(-q * T) * norm.cdf(d1)) / 365
            rho = K2 * T * np.exp(-r * T) * norm.cdf(d2) / 100 
        else:
            delta = -np.exp(-q * T) * norm.cdf(-d1)
            gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
            vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100
            theta = (-S * sigma * np.exp(-q * T) * norm.pdf(d1) / (2 * np.sqrt(T))
                    + r * K2 * np.exp(-r * T) * norm.cdf(-d2)
                    - q * S * np.exp(-q * T) * norm.cdf(-d1)) / 365
            rho = -K2 * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
        return {
            'price': price,
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho
        }
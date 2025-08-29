import numpy as np
from scipy.stats import norm
from .base_model import PricingModel

class BlackScholesModel(PricingModel):
    def calculate(self, S, K, T, r, sigma, option_type='call', q=0.0, **kwargs):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'call':
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        
        return {
            'price': price,
            'method': 'black_scholes',
            'option_style': 'european'
        }
    
    def calculate_greeks(self, S, K, T, r, sigma, option_type='call', q=0.0, **kwargs):
        if T <= 0:
            price = max(S - K, 0) if option_type == 'call' else max(K - S, 0)
            delta = 1.0 if (option_type == 'call' and S > K) or (option_type == 'put' and S < K) else 0.0
            return {'price': price, 'delta': delta, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0, 'rho': 0.0}
        
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        price = self.calculate(S, K, T, r, sigma, option_type)['price']
        
        if option_type == 'call':
            delta = np.exp(-q * T) * norm.cdf(d1)
            gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
            vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100
            theta = (-S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                     - r * K * np.exp(-r * T) * norm.cdf(d2) 
                     + q * S * np.exp(-q * T) * norm.cdf(d1)) / 365  
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100  
        else:
            delta = np.exp(-q * T) * (norm.cdf(d1) - 1)
            gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
            vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100
            theta = (-S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                     + r * K * np.exp(-r * T) * norm.cdf(-d2)
                     - q * S * np.exp(-q * T) * norm.cdf(-d1)) / 365
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
        return {
            'price': price,
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho
        }
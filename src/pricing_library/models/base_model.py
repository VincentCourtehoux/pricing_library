from abc import ABC, abstractmethod

class PricingModel(ABC):
    @abstractmethod
    def calculate(self, S, K, T, r, sigma, option_type, **kwargs):
        pass

    def calculate_greeks(self, S, K, T, r, sigma, option_type, **kwargs):
        base_price = self.calculate(S, K, T, r, sigma, option_type, **kwargs)['price']
        
        bump_S = 1.0
        price_up = self.calculate(S + bump_S, K, T, r, sigma, option_type, **kwargs)['price']
        price_down = self.calculate(S - bump_S, K, T, r, sigma, option_type, **kwargs)['price']
        delta = (price_up - price_down) / (2 * bump_S)
        gamma = (price_up - 2 * base_price + price_down) / (bump_S ** 2)
       
        bump_sigma = 0.01
        price_vega = self.calculate(S, K, T, r, sigma + bump_sigma, option_type, **kwargs)['price']
        vega = (price_vega - base_price) / bump_sigma
        
        bump_T = 1/365
        price_theta = self.calculate(S, K, max(T - bump_T, 1e-6), r, sigma, option_type, **kwargs)['price']
        theta = price_theta - base_price
      
        bump_r = 0.01
        price_rho = self.calculate(S, K, T, r + bump_r, sigma, option_type, **kwargs)['price']
        rho = (price_rho - base_price) / bump_r
        
        return {
            'price': base_price,
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho
        }
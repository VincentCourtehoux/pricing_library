import numpy as np
from ..utils.payoff import european_payoff
from .base_model import PricingModel

class BinomialModel(PricingModel):
    def calculate(self, S, K, T, r, sigma, option_type='call', **kwargs):
        n_steps = kwargs.get('n_steps', 100)
        option_style = kwargs.get('option_style', 'american')
        
        dt = T / n_steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(r * dt) - d) / (u - d)
        discount_factor = np.exp(-r * dt)
        
        stock_tree = self._build_stock_tree(S, u, d, n_steps)
        
        if option_style == 'european':
            price = self._price_european(stock_tree, K, option_type, p, discount_factor, n_steps)
        elif option_style == 'american':
            price = self._price_american(stock_tree, K, option_type, p, discount_factor, n_steps)
        else:
            raise ValueError(f'Unsupported option style: {option_style}')
        
        return {
            'price': price,
            'n_steps': n_steps,
            'method': 'binomial',
            'option_style': option_style
        }
    
    def _build_stock_tree(self, S0, u, d, n_steps):
        tree = np.zeros((n_steps + 1, n_steps + 1))
        for i in range(n_steps + 1):
            for j in range(i + 1):
                tree[j, i] = S0 * (u ** (i - j)) * (d ** j)
        return tree
    
    def _price_european(self, stock_tree, K, option_type, p, discount_factor, n_steps):
        option_values = european_payoff(stock_tree[:, n_steps], K, option_type)
        
        for step in range(n_steps - 1, -1, -1):
            new_values = np.zeros(step + 1)
            for node in range(step + 1):
                continuation = discount_factor * (
                    p * option_values[node] + (1 - p) * option_values[node + 1]
                )
                new_values[node] = continuation
            option_values = new_values
        
        return option_values[0]
    
    def _price_american(self, stock_tree, K, option_type, p, discount_factor, n_steps):
        option_tree = np.zeros_like(stock_tree)
        option_tree[:, n_steps] = european_payoff(stock_tree[:, n_steps], K, option_type)
        
        for step in range(n_steps - 1, -1, -1):
            for node in range(step + 1):
                continuation = discount_factor * (
                    p * option_tree[node, step + 1] + (1 - p) * option_tree[node + 1, step + 1]
                )
                exercise = european_payoff(np.array([stock_tree[node, step]]), K, option_type)[0]
                option_tree[node, step] = max(continuation, exercise)
        
        return option_tree[0, 0]
import numpy as np

def intrinsic_value(prices, K, option_type='call'):
    if option_type == 'call':
        return np.maximum(prices - K, 0)
    else:
        return np.maximum(K - prices, 0)

european_payoff = intrinsic_value
    
def asian_payoff(prices, K, option_type='call', averaging_type='arithmetic'):
    if averaging_type == 'arithmetic':
        average_prices = np.mean(prices, axis=1)
    elif averaging_type == 'geometric':
        average_prices = np.exp(np.mean(np.log(prices), axis=1))
    else:
        raise ValueError(f'Unsupported averaging type: {averaging_type}')
    
    if option_type == 'call':
        return np.maximum(average_prices - K, 0)
    else:
        return np.maximum(K - average_prices, 0)
    
def gap_payoff(S, K1, K2, option_type='call'):
    if option_type == 'call':
        return max(S - K2, 0) if S > K1 else 0
    elif option_type == 'put':
        return max(K2 - S, 0) if S < K1 else 0
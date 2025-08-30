from pricing_library.core.pricing_service import PricingService
import numpy as np
service = PricingService()

params = {
    'S': 100,
    'K': 100,
    'T': 1,
    'r': 0.05,
    'sigma': 0.2,
    'option_type': 'call',
    'K1': 110,
    'K2': 105,
    'option_style': 'gap',
    'seed': 42
}

gap_price = service.price_option(params, 'monte_carlo')['price']
print(f'Asian premium: {gap_price}')
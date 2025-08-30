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
    'option_style': 'asian',   # <-- ici tu veux une asiatique
    'seed': 42,
    't_today': 0.2,
    'observed_values': [102, 98],   # <-- par ex. 2 fixings connus
    'monitoring_dates': [0.1, 0.2, 0.5, 1]  # toutes les dates
}
asian_price = service.price_option(params, 'monte_carlo')['price']
print(f'Asian premium: {asian_price}')

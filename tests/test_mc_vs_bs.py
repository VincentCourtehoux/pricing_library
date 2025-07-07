import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.models.vanilla_options.european_options.black_scholes.pricing_scalar import BlackScholesScalar
bs = BlackScholesScalar()
from core.models.vanilla_options.european_options.monte_carlo import mc_eu_premium

def test_put_call_value():
    params = {
        'S0': 100,
        'K': 100,
        'T': 1.0,
        'r': 0.05,
        'sigma': 0.2,
        'q': 0.02,
        'N': 100,
        'nb_paths': 50000
    }
    
    for option_type in ['call', 'put']:
        result = mc_eu_premium(**params, option_type=option_type, return_all=True, seed=42)
        assert result['confidence_interval'][0] <= result['black_scholes_price'] <= result['confidence_interval'][1]

if __name__ == "__main__":
    pytest.main(["tests/test_mc_vs_bs.py"])
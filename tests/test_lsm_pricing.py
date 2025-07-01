import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.models.american.longstaff_schwartz.pricing import longstaff_schwartz_american
from core.models.european.black_scholes.pricing_scalar import BlackScholesScalar
bs = BlackScholesScalar()

def test_american_vs_european():
    params = {
        'S0': 36,
        'K': 40,
        'r': 0.06,
        'sigma': 0.2,
        'T': 1.0,
        'q': 0.0,
        'N': 50,
        'nb_paths': 50000,
        'seed': 42
    }

    params2 = {
        'S': 36,
        'K': 40,
        'T': 1.0,
        'r': 0.06,
        'sigma': 0.2,
        'q': 0.0,
    }

    for option_type in ['call', 'put']:
        american_price = longstaff_schwartz_american(**params, option_type=option_type)
        european_price = bs.premium(**params2, option_type=option_type)

        tolerance = 1e-2
        msg = f"Erreur : {option_type} américaine ({american_price:.4f}) < européenne ({european_price:.4f})"

        assert american_price + tolerance >= european_price, msg

if __name__ == "__main__":
    pytest.main(["tests/test_lsm_pricing.py"])
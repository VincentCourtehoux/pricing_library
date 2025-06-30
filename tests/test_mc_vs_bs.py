import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from core.models.european.black_scholes.pricing_scalar import BlackScholesScalar
bs = BlackScholesScalar()
from core.models.utils.monte_carlo_pricing import MonteCarloPricing
mc = MonteCarloPricing()


def test_call_put_value():
    
    results = mc.monte_carlo_pricing(100, 100, 1, 0.05, 0.2, 0.02, 252, 100000, "call", True)
    mc_price = results["price"]
    ci_low, ci_high = results["confidence_interval"]

    bs_price = bs.premium(100, 100, 1, 0.05, 0.2, 0.02, "call")
    assert ci_low <= bs_price <= ci_high, f"BS price {bs_price} not in [{ci_low}, {ci_high}]"

if __name__ == "__main__":
    pytest.main(["tests/test_mc_vs_bs.py"])
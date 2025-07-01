import numpy as np
import sys
import os
import pytest 

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.models.american.binomial_tree import BinomialTreeAmerican

def test_american_call_price():
    
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.2
    n = 50
    
    american_call = BinomialTreeAmerican(S0, K, T, r, sigma, n, 'call')
    option_price = american_call.price_option()
    expected_price = 10.45
    
    assert abs(option_price - expected_price) < 0.1, f"Expected price ~{expected_price}, got {option_price:.4f}"

def test_american_put_price():
    
    S0 = 80.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.3
    n = 3
    
    american_put = BinomialTreeAmerican(S0, K, T, r, sigma, n, 'put')
    option_price = american_put.price_option()
    intrinsic_value = max(K - S0, 0)
    
    assert option_price >= intrinsic_value, f"Option price {option_price:.4f} should be >= intrinsic value {intrinsic_value}"
    assert option_price <= K, f"Put option price {option_price:.4f} should not exceed strike price {K}"

if __name__ == "__main__":
    pytest.main(["tests/test_us_binomial_tree.py"])
import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.models.european.black_scholes.vol_scalar import implied_volatility_scalar
from core.models.european.black_scholes.pricing_scalar import BlackScholesScalar
bs = BlackScholesScalar()

def test_implied_volatility_recovery():
    """
    Test that the implied volatility function can correctly recover the original
    volatility used to generate the option price.
    """
    S = 100
    K = 100
    T = 1
    r = 0.05
    true_vol = 0.2
    q = 0.02
    option_type = "call"

    price = bs.premium(S, K, T, r, true_vol, q, option_type)
    iv = implied_volatility_scalar(price, S, K, T, r, q, option_type)

    assert np.allclose(iv, true_vol, atol=1e-4)

def test_implied_volatility_zero_maturity():
    """
    Test that the implied volatility function handles the edge case where
    time to maturity is zero. Expected to return NaN or an error-safe value.
    """
    S = 100
    K = 100
    T = 0
    r = 0.05
    q = 0.02
    price = 5.0
    option_type = "call"

    iv = implied_volatility_scalar(price, S, K, T, r, q, option_type)
    assert np.isnan(iv)

def test_implied_volatility_deep_itm():
    """
    Test that the implied volatility function can accurately recover the volatility
    for a deep in-the-money call option (S >> K).
    """
    S = 150
    K = 100
    T = 1
    r = 0.05
    true_vol = 0.3
    q = 0.02
    option_type = "call"

    price = bs.premium(S, K, T, r, true_vol, q, option_type)
    iv = implied_volatility_scalar(price, S, K, T, r, q, option_type)

    assert np.allclose(iv, true_vol, atol=1e-3)

def test_implied_volatility_invalid_price():
    """
    Test that the implied volatility function handles an invalid input price
    (e.g., too high to correspond to any realistic volatility). Should return NaN.
    """
    S = 100
    K = 100
    T = 1
    r = 0.05
    q = 0.02
    price = 9999
    option_type = "call"

    iv = implied_volatility_scalar(price, S, K, T, r, q, option_type)
    assert np.isnan(iv)

if __name__ == "__main__":
    pytest.main(["tests/test_bs_vol_scalar.py"])
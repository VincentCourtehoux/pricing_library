import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.models.vanilla.european.black_scholes.vol_vectorized import implied_volatility_vectorized
from core.models.vanilla.european.black_scholes.pricing_vectorized import BlackScholesVectorized
bs = BlackScholesVectorized()

def test_implied_volatility_recovery():
    """
    Test that the implied volatility function can correctly recover the original
    volatility used to generate the option price.
    """
    S = np.array([100])
    K = np.array([100])
    T = np.array([1])
    r = np.array([0.05])
    true_vol = np.array([0.2])
    q = np.array([0.02])
    option_type = np.array(["call"])

    price = bs.premium(S, K, T, r, true_vol, q, option_type)
    iv = implied_volatility_vectorized(price, S, K, T, r, q, option_type)

    assert np.allclose(iv, true_vol, atol=1e-4)

def test_implied_volatility_zero_maturity():
    """
    Test that the implied volatility function handles the edge case where
    time to maturity is zero. Expected to return NaN or an error-safe value.
    """
    S = np.array([100])
    K = np.array([100])
    T = np.array([0])
    r = np.array([0.05])
    q = np.array([0.02])
    price = np.array([5.0])
    option_type = np.array(["call"])

    iv = implied_volatility_vectorized(price, S, K, T, r, q, option_type)
    assert np.isnan(iv[0])

def test_implied_volatility_deep_itm():
    """
    Test that the implied volatility function can accurately recover the volatility
    for a deep in-the-money call option (S >> K).
    """
    S = np.array([150])
    K = np.array([100])
    T = np.array([1])
    r = np.array([0.05])
    true_vol = np.array([0.3])
    q = np.array([0.02])
    option_type = np.array(["call"])

    price = bs.premium(S, K, T, r, true_vol, q, option_type)
    iv = implied_volatility_vectorized(price, S, K, T, r, q, option_type)

    assert np.allclose(iv, true_vol, atol=1e-3)

def test_implied_volatility_invalid_price():
    """
    Test that the implied volatility function handles an invalid input price
    (e.g., too high to correspond to any realistic volatility). Should return NaN.
    """
    S = np.array([100])
    K = np.array([100])
    T = np.array([1])
    r = np.array([0.05])
    q = np.array([0.02])
    price = np.array([9999.0])
    option_type = np.array(["call"])

    iv = implied_volatility_vectorized(price, S, K, T, r, q, option_type)
    assert np.isnan(iv[0])

if __name__ == "__main__":
    pytest.main(["tests/test_bs_vol_vectorized.py"])
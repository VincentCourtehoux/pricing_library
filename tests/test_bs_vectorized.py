import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.models.vanilla_options.european_options.black_scholes.pricing_vectorized import BlackScholesVectorized
bs = BlackScholesVectorized()

def test_premium_call_put_known_value():
    S = np.array([100, 100])
    K = np.array([100, 100])
    T = np.array([1, 1])
    r = np.array([0.05, 0.05])
    q = np.array([0.00, 0.00])
    sigma = np.array([0.2, 0.2])
    option_type = np.array(["call", "put"])

    price = bs.bs_eu_vectorized_premium(S, K, T, r, sigma, q, option_type)
    assert np.allclose(price, [10.4506, 5.5735], atol=1e-4)
    assert np.allclose(price[0] - price[1], S[0] * np.exp(-q[0] * T[0]) - K[0] * np.exp(-r[0] * T[0]), atol=1e-4)

def test_zero_maturity_price():
    S = np.array([100])
    K = np.array([100])
    T = np.array([0])
    r = np.array([0.05])
    q = np.array([0.02])
    sigma = np.array([0.2])
    call = np.array(["call"])
    put = np.array(["put"])

    assert bs.bs_eu_vectorized_premium(S, K, T, r, sigma, q, call) == pytest.approx([max(S[0] - K[0], 0)])
    assert bs.bs_eu_vectorized_premium(S, K, T, r, sigma, q, put) == pytest.approx([max(K[0] - S[0], 0)])

def test_zero_volatility_price():
    S = np.array([100])
    K = np.array([100])
    T = np.array([1])
    r = np.array([0.05])
    q = np.array([0.02])
    sigma = np.array([0.0])
    call = np.array(["call"])
    put = np.array(["put"])

    call_price = bs.bs_eu_vectorized_premium(S, K, T, r, sigma, q, call)
    put_price = bs.bs_eu_vectorized_premium(S, K, T, r, sigma, q, put)

    call_intrinsic = max(S[0] * np.exp(-q[0] * T[0]) - K[0] * np.exp(-r[0] * T[0]), 0)
    put_intrinsic = max(K[0] * np.exp(-r[0] * T[0]) - S[0] * np.exp(-q[0] * T[0]), 0)

    assert call_price == pytest.approx([call_intrinsic])
    assert put_price == pytest.approx([put_intrinsic])

def test_invariance_scale():
    S = np.array([100])
    K = np.array([100])
    T = np.array([1])
    r = np.array([0.05])
    q = np.array([0.02])
    sigma = np.array([0.2])
    call = np.array(["call"])

    price1 = bs.bs_eu_vectorized_premium(S, K, T, r, sigma, q, call)
    price2 = bs.bs_eu_vectorized_premium(S * 10, K * 10, T, r, sigma, q, call)

    assert np.allclose(price2, price1 * 10, atol=1e-4)

def test_delta_bounds():
    S = np.array([100, 100])
    K = np.array([90, 110])
    T = np.array([1, 1])
    r = np.array([0.05, 0.05])
    q = np.array([0.02, 0.02])
    sigma = np.array([0.2, 0.2])
    option_type = np.array(["call", "put"])

    delta = bs.delta(S, K, T, r, sigma, q, option_type)
    assert delta[0] > 0 and delta[0] < 1
    assert delta[1] < 0 and delta[1] > -1

def test_gamma_positive():
    S = np.array([90, 100, 110])
    K = np.array([100, 100, 100])
    T = np.array([1, 1, 1])
    r = np.array([0.05, 0.05, 0.05])
    q = np.array([0.02, 0.02, 0.02])
    sigma = np.array([0.2, 0.2, 0.2])

    gamma = bs.gamma(S, K, T, r, sigma, q)
    assert np.all(gamma >= 0)

def test_vega_positive():
    S = np.array([90, 100, 110])
    K = np.array([100, 100, 100])
    T = np.array([1, 1, 1])
    r = np.array([0.05, 0.05, 0.05])
    q = np.array([0.02, 0.02, 0.02])
    sigma = np.array([0.2, 0.2, 0.2])

    vega = bs.vega(S, K, T, r, sigma, q)
    assert np.all(vega >= 0)

def test_theta_sign():
    S = np.array([100])
    K = np.array([100])
    T = np.array([1])
    r = np.array([0.05])
    q = np.array([0.02])
    sigma = np.array([0.2])
    call = np.array(["call"])
    put = np.array(["put"])

    theta_call = bs.theta(S, K, T, r, sigma, q, call)
    theta_put = bs.theta(S, K, T, r, sigma, q, put)
    assert theta_call < 0
    assert theta_put < 0

def test_rho_sign():
    S = np.array([100])
    K = np.array([100])
    T = np.array([1])
    r = np.array([0.05])
    q = np.array([0.02])
    sigma = np.array([0.2])
    call = np.array(["call"])
    put = np.array(["put"])

    rho_call = bs.rho(S, K, T, r, sigma, q, call)
    rho_put = bs.rho(S, K, T, r, sigma, q, put)
    assert rho_call > 0
    assert rho_put < 0

if __name__ == "__main__":
    pytest.main(["tests/test_bs_vectorized.py"])

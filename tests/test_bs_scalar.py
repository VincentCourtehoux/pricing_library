import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.models.vanilla.european.black_scholes.pricing_scalar import BlackScholesScalar
bs = BlackScholesScalar()

def test_premium_call_put_known_value():
    S = 100
    K = 100
    T = 1
    r = 0.05
    q = 0.00
    sigma = 0.2
    call_price = bs.premium(S, K, T, r, sigma, q, "call")
    put_price = bs.premium(S, K, T, r, sigma, q, "put")

    # Reference value : call = 10.4506; put = 5.5735
    assert np.allclose(call_price, 10.4506, atol=1e-4)
    assert np.allclose(put_price, 5.5735, atol=1e-4)
    # Put-call parity with dividends
    assert np.allclose(call_price - put_price, S * np.exp(-q * T) - K * np.exp(-r * T), atol=1e-4)

def test_zero_maturity_price():
    S = 100
    K = 100
    T = 0
    r = 0.05
    q = 0.02
    sigma = 0.2

    assert bs.premium(S, K, T, r, sigma, q, "call") == pytest.approx(max(S - K, 0))
    assert bs.premium(S, K, T, r, sigma, q, "put") == pytest.approx(max(K - S, 0))

def test_zero_volatility_price():
    S = 100
    K = 100
    T = 1
    r = 0.05
    q = 0.02
    sigma = 0

    call_price = bs.premium(S, K, T, r, sigma, q, "call")
    put_price = bs.premium(S, K, T, r, sigma, q, "put")

    call_intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0)
    put_intrinsic = max(K * np.exp(-r * T) - S * np.exp(-q * T), 0)

    assert call_price == pytest.approx(call_intrinsic)
    assert put_price == pytest.approx(put_intrinsic)

def test_invariance_scale():
    S = 100
    K = 100
    T = 1
    r = 0.05
    q = 0.02
    sigma = 0.2

    price1 = bs.premium(S, K, T, r, sigma, q, "call")
    price2 = bs.premium(S * 10, K * 10, T, r, sigma, q, "call")
    assert np.allclose(price2, price1 * 10, atol=1e-4)

def test_delta_bounds():
    S = np.array([100, 100])
    K = np.array([90, 110])
    T = np.array([1, 1])
    r = np.array([0.05, 0.05])
    q = np.array([0.02, 0.02])
    sigma = np.array([0.2, 0.2])

    delta_call = bs.delta(S[0], K[0], T[0], r[0], sigma[0], q[0], "call")
    delta_put = bs.delta(S[1], K[1], T[1], r[1], sigma[1], q[1], "put")
    assert 0 < delta_call < 1
    assert -1 < delta_put < 0

def test_gamma_positive():
    S = np.array([90, 100, 110])
    K = np.array([100, 100, 100])
    T = np.array([1, 1, 1])
    r = np.array([0.05, 0.05, 0.05])
    q = np.array([0.02, 0.02, 0.02])
    sigma = np.array([0.2, 0.2, 0.2])

    gamma_1 = bs.gamma(S[0], K[0], T[0], r[0], sigma[0], q[0])
    gamma_2 = bs.gamma(S[1], K[1], T[1], r[1], sigma[1], q[1])
    gamma_3 = bs.gamma(S[2], K[2], T[2], r[2], sigma[2], q[2])
    assert gamma_1 > 0 and gamma_2 > 0 and gamma_3 > 0

def test_vega_positive():
    S = np.array([90, 100, 110])
    K = np.array([100, 100, 100])
    T = np.array([1, 1, 1])
    r = np.array([0.05, 0.05, 0.05])
    q = np.array([0.02, 0.02, 0.02])
    sigma = np.array([0.2, 0.2, 0.2])

    vega_1 = bs.vega(S[0], K[0], T[0], r[0], sigma[0], q[0])
    vega_2 = bs.vega(S[1], K[1], T[1], r[1], sigma[1], q[1])
    vega_3 = bs.vega(S[2], K[2], T[2], r[2], sigma[2], q[2])
    assert vega_1 > 0 and vega_2 > 0 and vega_3 > 0

def test_theta_sign():
    S = 100
    K = 100
    T = 1
    r = 0.05
    q = 0.02
    sigma = 0.2

    theta_call = bs.theta(S, K, T, r, sigma, q, "call")
    theta_put = bs.theta(S, K, T, r, sigma, q, "put")
    assert theta_call < 0
    assert theta_put < 0

def test_rho_sign():
    S = 100
    K = 100
    T = 1
    r = 0.05
    q = 0.02
    sigma = 0.2

    rho_call = bs.rho(S, K, T, r, sigma, q, "call")
    rho_put = bs.rho(S, K, T, r, sigma, q, "put")
    assert rho_call > 0
    assert rho_put < 0

if __name__ == "__main__":
    pytest.main(["tests/test_bs_scalar.py"])

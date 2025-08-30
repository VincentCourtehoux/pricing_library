import pytest
import numpy as np
from pricing_library.models.black_scholes import BlackScholesModel

@pytest.fixture
def bs_model():
    return BlackScholesModel()

def test_call_price(bs_model):
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    res = bs_model.calculate(S, K, T, r, sigma, option_type='call')
    price_call = res['price']
    ref_call = 10.45
    assert abs(price_call - ref_call) / ref_call < 0.01

def test_put_price(bs_model):
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    res = bs_model.calculate(S, K, T, r, sigma, option_type='put')
    price_put = res['price']
    ref_put = 5.57
    assert abs(price_put - ref_put) / ref_put < 0.01

def test_put_call_parity(bs_model):
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    call = bs_model.calculate(S, K, T, r, sigma, option_type='call')['price']
    put = bs_model.calculate(S, K, T, r, sigma, option_type='put')['price']
    assert abs((call - put) - (S - K*np.exp(-r*T))) < 0.01

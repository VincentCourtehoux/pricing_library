import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.models.exotic_options.barrier_options.mc_barrier_pricing import mc_barrier_premium

def test_vanilla_call_approximation():
    np.random.seed(42)
    price = mc_barrier_premium(
        100, 100, 1, 0.05, 0.2, 0, 10, 252, 100000, 'call', 'down-out'
    )
    bs_call = 10.45
    assert abs(price - bs_call) < 1.0
    
def test_up_out_cheaper_than_vanilla():
    np.random.seed(42)
    up_out_price = mc_barrier_premium(
        100, 100, 1, 0.05, 0.2, 0, 110, 252, 50000, 'call', 'up-out'
    )
    vanilla_approx = mc_barrier_premium(
        100, 100, 1, 0.05, 0.2, 0, 10, 252, 50000, 'call', 'down-out'
    )
    assert up_out_price < vanilla_approx

def test_down_out_put_cheaper_than_vanilla():
    np.random.seed(42)
    down_out_price = mc_barrier_premium(
        100, 100, 1, 0.05, 0.2, 0, 80, 252, 50000, 'put', 'down-out'
    )
    vanilla_approx = mc_barrier_premium(
        100, 100, 1, 0.05, 0.2, 0, 200, 252, 50000, 'put', 'up-out'
    )
    assert down_out_price < vanilla_approx

def test_barrier_far_away_equals_vanilla():
    np.random.seed(42)
    far_barrier_price = mc_barrier_premium(
        100, 100, 1, 0.05, 0.2, 0, 200, 252, 50000, 'call', 'up-out'
    )
    vanilla_approx = mc_barrier_premium(
        100, 100, 1, 0.05, 0.2, 0, 10, 252, 50000, 'call', 'down-out'
    )
    assert abs(far_barrier_price - vanilla_approx) < 0.5

def test_zero_volatility_deterministic():
    price = mc_barrier_premium(
        100, 90, 1, 0.05, 0.0, 0, 110, 252, 1000, 'call', 'up-out'
    )
    expected = (100 * np.exp(0.05) - 90) * np.exp(-0.05)
    assert abs(price - expected) < 0.01

def test_deep_otm_near_zero():
    price = mc_barrier_premium(
        100, 150, 1, 0.05, 0.2, 0, 110, 252, 50000, 'call', 'up-out'
    )
    assert price < 1.0

def test_barrier_in_vs_out_complement():
    np.random.seed(42)
    up_in_price = mc_barrier_premium(
        100, 100, 1, 0.05, 0.2, 0, 110, 252, 100000, 'call', 'up-in'
    )
    np.random.seed(42)
    up_out_price = mc_barrier_premium(
        100, 100, 1, 0.05, 0.2, 0, 110, 252, 100000, 'call', 'up-out'
    )
    np.random.seed(42)
    vanilla_approx = mc_barrier_premium(
        100, 100, 1, 0.05, 0.2, 0, 10, 252, 100000, 'call', 'down-out'
    )
    assert abs((up_in_price + up_out_price) - vanilla_approx) < 0.5

if __name__ == "__main__":
    pytest.main(["tests/test_mc_barrier_pricing.py"])
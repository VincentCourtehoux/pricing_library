import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.models.utils.statistics import confidence_interval
from core.models.utils.path_simulation import simulate_gbm
from core.models.american.longstaff_schwartz.pricing import longstaff_schwartz
from core.models.european.black_scholes.pricing_scalar import BlackScholesScalar
bs = BlackScholesScalar()

def test_lsm_put_american_more_than_european():

    S0 = 1
    K = 1.1
    r = 0.02
    sigma = 0.15
    T = 0.5
    q = 0.4
    N = 50 
    nb_paths = 20000 
    option_type = "put"
    dt = T/N

    S = simulate_gbm(S0, r, sigma, T, q, N, nb_paths, seed=42)
    price = longstaff_schwartz(S, K, r, dt, degree=3)
    bs_price = bs.premium(S0, K, T, r, sigma, q, option_type)

    assert price == bs_price, f"Put américain ({price}) doit être > européen ({bs_price})"

if __name__ == "__main__":
    pytest.main(["tests/test_lsm.py"])
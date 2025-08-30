import pytest
from pricing_library.models.black_scholes import BlackScholesModel
from pricing_library.models.black_scholes_barrier import BlackScholesBarrierModel

euro_model = BlackScholesModel()
barrier_model = BlackScholesBarrierModel()

S0 = 100
K = 100
r = 0.05
sigma = 0.2
T = 1
barrier_down = 90
barrier_up = 110

barriers = [
    ("down-and-out", barrier_down),
    ("down-and-in", barrier_down),
    ("up-and-out", barrier_up),
    ("up-and-in", barrier_up)
]

options = ["call", "put"]

@pytest.mark.parametrize("barrier,barrier_level", barriers)
@pytest.mark.parametrize("option_type", options)
def test_barrier_option_consistency(barrier, barrier_level, option_type):
    price = barrier_model.calculate(
        S=S0, K=K, r=r, sigma=sigma, T=T,
        option_type=option_type, barrier_type=barrier, barrier_level=barrier_level
    )['price']
    
    euro_price = euro_model.calculate(
        S=S0, K=K, r=r, sigma=sigma, T=T, option_type=option_type
    )['price']
    
    assert price >= 0, f"Prix négatif pour {barrier} {option_type}"
    assert price <= euro_price, f"Prix supérieur à l'européenne pour {barrier} {option_type}"

@pytest.mark.parametrize("option_type", options)
def test_in_out_sum_equals_european(option_type):
    down_in = barrier_model.calculate(
        S=S0, K=K, r=r, sigma=sigma, T=T,
        option_type=option_type, barrier_type="down-and-in", barrier_level=barrier_down
    )['price']
    down_out = barrier_model.calculate(
        S=S0, K=K, r=r, sigma=sigma, T=T,
        option_type=option_type, barrier_type="down-and-out", barrier_level=barrier_down
    )['price']
    
    up_in = barrier_model.calculate(
        S=S0, K=K, r=r, sigma=sigma, T=T,
        option_type=option_type, barrier_type="up-and-in", barrier_level=barrier_up
    )['price']
    up_out = barrier_model.calculate(
        S=S0, K=K, r=r, sigma=sigma, T=T,
        option_type=option_type, barrier_type="up-and-out", barrier_level=barrier_up
    )['price']
    
    euro_price = euro_model.calculate(
        S=S0, K=K, r=r, sigma=sigma, T=T, option_type=option_type
    )['price']
    
    assert abs(down_in + down_out - euro_price) < 1e-8, f"Down in+out != européen pour {option_type}"
    assert abs(up_in + up_out - euro_price) < 1e-8, f"Up in+out != européen pour {option_type}"

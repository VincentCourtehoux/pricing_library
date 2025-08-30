import pytest
from pricing_library.models.least_squares_mc import LeastSquaresMC
from pricing_library.models.black_scholes import BlackScholesModel

lsm_model = LeastSquaresMC()
euro_model = BlackScholesModel()

S0 = 100
K = 100
r = 0.05
sigma = 0.2
T = 1
n_paths = 10000
n_steps = 100

options = ["call", "put"]

@pytest.mark.parametrize("option_type", options)
def test_american_vs_european(option_type):
    american_price = lsm_model.calculate(
        S=S0, K=K, r=r, sigma=sigma, T=T,
        option_type=option_type, n_paths=n_paths, 
        n_steps=n_steps
    )['price']
    
    european_price = euro_model.calculate(
        S=S0, K=K, r=r, sigma=sigma, T=T,
        option_type=option_type
    )['price']
    
    if option_type == "put":
        assert american_price >= european_price, (
            f"Prix américain {american_price} < prix européen {european_price} pour un put"
        )
    else:
        assert abs(american_price - european_price) / european_price < 0.01, (
            f"Prix américain {american_price} trop différent du prix européen {european_price} pour un call"
        )

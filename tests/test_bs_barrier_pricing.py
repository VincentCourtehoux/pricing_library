import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.models.exotic_options.barrier_options.bs_barrier_pricing import bs_barrier_premium
from core.models.vanilla_options.european_options.black_scholes.pricing_scalar import BlackScholesScalar

bs = BlackScholesScalar()

class TestBarrierOptions:
    
    def test_down_and_out_call_basic(self):
        S = 100
        K = 100
        T = 1.0
        r = 0.05
        sigma = 0.2
        q = 0.0
        H = 95
        rebate = 0.0
        
        price = bs_barrier_premium(S, K, T, r, sigma, q, H, "call", "down-out")
        
        assert price > 0
        assert price < bs.bs_eu_scalar_premium(S, K, T, r, sigma, q, "call")
        
    def test_down_and_out_call_at_barrier(self):
        S = 95
        K = 100
        T = 1.0
        r = 0.05
        sigma = 0.2
        q = 0.0
        H = 95
        rebate = 0.0
        
        price = bs_barrier_premium(S, K, T, r, sigma, q, H, "call", "down-out")
        
        assert price == 0
        
    def test_down_and_out_call_below_barrier(self):
        S = 90
        K = 100
        T = 1.0
        r = 0.05
        sigma = 0.2
        q = 0.0
        H = 95
        rebate = 0.0
        
        price = bs_barrier_premium(S, K, T, r, sigma, q, H, "call", "down-out")
        
        assert price == 0
        
    def test_down_and_in_call_basic(self):
        S = 105
        K = 100
        T = 1.0
        r = 0.05
        sigma = 0.2
        q = 0.0
        H = 100
        rebate = 0.0
        
        price_di = bs_barrier_premium(S, K, T, r, sigma, q, H, "call", "down-in")
        price_do = bs_barrier_premium(S, K, T, r, sigma, q, H, "call", "down-out")
        european_price = bs.bs_eu_scalar_premium(S, K, T, r, sigma, q, "call")
        
        assert abs(price_di + price_do - european_price) < 1e-10
        
    def test_up_and_out_call_basic(self):
        S = 100
        K = 100
        T = 1.0
        r = 0.05
        sigma = 0.2
        q = 0.0
        H = 105
        rebate = 0.0
        
        price = bs_barrier_premium(S, K, T, r, sigma, q, H, "call", "up-out")
        
        assert price >= 0
        assert price <= bs.bs_eu_scalar_premium(S, K, T, r, sigma, q, "call")
        
    def test_up_and_out_call_at_barrier(self):
        S = 105
        K = 100
        T = 1.0
        r = 0.05
        sigma = 0.2
        q = 0.0
        H = 105
        rebate = 0.0
        
        price = bs_barrier_premium(S, K, T, r, sigma, q, H, "call", "up-out")
        
        assert price == 0
        
    def test_up_and_in_call_basic(self):
        S = 100
        K = 100
        T = 1.0
        r = 0.05
        sigma = 0.2
        q = 0.0
        H = 105
        rebate = 0.0
        
        price_ui = bs_barrier_premium(S, K, T, r, sigma, q, H, "call", "up-in")
        price_uo = bs_barrier_premium(S, K, T, r, sigma, q, H, "call", "up-out")
        european_price = bs.bs_eu_scalar_premium(S, K, T, r, sigma, q, "call")
        
        assert abs(price_ui + price_uo - european_price) < 1e-10
        
    def test_down_and_out_put_basic(self):
        S = 100
        K = 100
        T = 1.0
        r = 0.05
        sigma = 0.2
        q = 0.0
        H = 95
        rebate = 0.0
        
        price = bs_barrier_premium(S, K, T, r, sigma, q, H, "put", "down-out")
        
        assert price >= 0
        assert price <= bs.bs_eu_scalar_premium(S, K, T, r, sigma, q, "put")
        
    def test_down_and_out_put_at_barrier(self):
        S = 95
        K = 100
        T = 1.0
        r = 0.05
        sigma = 0.2
        q = 0.0
        H = 95
        rebate = 0.0
        
        price = bs_barrier_premium(S, K, T, r, sigma, q, H, "put", "down-out")
        
        assert price == 0
        
    def test_down_and_in_put_basic(self):
        S = 100
        K = 100
        T = 1.0
        r = 0.05
        sigma = 0.2
        q = 0.0
        H = 95
        rebate = 0.0
        
        price_di = bs_barrier_premium(S, K, T, r, sigma, q, H, "put", "down-in")
        price_do = bs_barrier_premium(S, K, T, r, sigma, q, H, "put", "down-out")
        european_price = bs.bs_eu_scalar_premium(S, K, T, r, sigma, q, "put")
        
        assert abs(price_di + price_do - european_price) < 1e-10
        
    def test_up_and_out_put_basic(self):
        S = 100
        K = 100
        T = 1.0
        r = 0.05
        sigma = 0.2
        q = 0.0
        H = 105
        rebate = 0.0
        
        price = bs_barrier_premium(S, K, T, r, sigma, q, H, "put", "up-out")
        
        assert price >= 0
        assert price <= bs.bs_eu_scalar_premium(S, K, T, r, sigma, q, "put")
        
    def test_up_and_out_put_at_barrier(self):
        S = 105
        K = 100
        T = 1.0
        r = 0.05
        sigma = 0.2
        q = 0.0
        H = 105
        rebate = 0.0
        
        price = bs_barrier_premium(S, K, T, r, sigma, q, H, "put", "up-out")
        
        assert price == 0
        
    def test_up_and_in_put_basic(self):
        S = 100
        K = 100
        T = 1.0
        r = 0.05
        sigma = 0.2
        q = 0.0
        H = 105
        rebate = 0.0
        
        price_ui = bs_barrier_premium(S, K, T, r, sigma, q, H, "put", "up-in")
        price_uo = bs_barrier_premium(S, K, T, r, sigma, q, H, "put", "up-out")
        european_price = bs.bs_eu_scalar_premium(S, K, T, r, sigma, q, "put")
        
        assert abs(price_ui + price_uo - european_price) < 1e-10
        
    def test_dividend_yield_effect(self):
        S = 100
        K = 100
        T = 1.0
        r = 0.05
        sigma = 0.2
        q1 = 0.0
        q2 = 0.02
        H = 95
        rebate = 0.0
        
        price_no_div = bs_barrier_premium(S, K, T, r, sigma, q1, H, "call", "down-out")
        price_with_div = bs_barrier_premium(S, K, T, r, sigma, q2, H, "call", "down-out")
        
        assert price_with_div < price_no_div
        
    def test_barrier_very_low_down_and_out_call(self):
        S = 100
        K = 100
        T = 1.0
        r = 0.05
        sigma = 0.2
        q = 0.0
        H = 1
        rebate = 0.0
        
        price = bs_barrier_premium(S, K, T, r, sigma, q, H, "call", "down-out")
        european_price = bs.bs_eu_scalar_premium(S, K, T, r, sigma, q, "call")
        
        assert abs(price - european_price) < 1e-6
        
    def test_barrier_very_high_up_and_out_call(self):
        S = 100
        K = 100
        T = 1.0
        r = 0.05
        sigma = 0.2
        q = 0.0
        H = 1000
        rebate = 0.0
        
        price = bs_barrier_premium(S, K, T, r, sigma, q, H, "call", "up-out")
        european_price = bs.bs_eu_scalar_premium(S, K, T, r, sigma, q, "call")
        
        assert abs(price - european_price) < 1e-6
        
    def test_time_to_expiry_zero(self):
        S = 100
        K = 100
        T = 1e-25
        r = 0.05
        sigma = 0.2
        q = 0.0
        H = 95
        rebate = 0.0
        
        price = bs_barrier_premium(S, K, T, r, sigma, q, H, "call", "down-out")
        expected = max(0, S - K) if S > H else 0
        
        assert abs(price - expected) < 1e-10
        
    def test_put_call_parity_down_barriers(self):
        S = 100
        K = 100
        T = 1.0
        r = 0.05
        sigma = 0.2
        q = 0.02
        H = 95
        rebate = 0.0
        
        call_do = bs_barrier_premium(S, K, T, r, sigma, q, H, "call", "down-out")
        put_do = bs_barrier_premium(S, K, T, r, sigma, q, H, "put", "down-out")
        
        call_di = bs_barrier_premium(S, K, T, r, sigma, q, H, "call", "down-in")
        put_di = bs_barrier_premium(S, K, T, r, sigma, q, H, "put", "down-in")
        
        european_parity = S * np.exp(-q * T) - K * np.exp(-r * T)
        barrier_parity_out = call_do - put_do
        barrier_parity_in = call_di - put_di
        
        assert abs(barrier_parity_out + barrier_parity_in - european_parity) < 1e-10
        
    def test_high_volatility_convergence(self):
        S = 100
        K = 100
        T = 1.0
        r = 0.05
        sigma = 5.0
        q = 0.0
        H = 95
        rebate = 0.0
        
        price_do = bs_barrier_premium(S, K, T, r, sigma, q, H, "call", "down-out")
        price_di = bs_barrier_premium(S, K, T, r, sigma, q, H, "call", "down-in")
        
        assert price_do >= 0
        assert price_di >= 0
        assert price_do + price_di > 0
        
    def test_interest_rate_zero(self):
        S = 100
        K = 100
        T = 1.0
        r = 0.0
        sigma = 0.2
        q = 0.0
        H = 95
        rebate = 0.0
        
        price = bs_barrier_premium(S, K, T, r, sigma, q, H, "call", "down-out")
        
        assert price >= 0
        assert price <= bs.bs_eu_scalar_premium(S, K, T, r, sigma, q, "call")
        
    def test_strike_equal_barrier_down_and_out_call(self):
        S = 100
        K = 95
        T = 1.0
        r = 0.05
        sigma = 0.2
        q = 0.0
        H = 95
        rebate = 0.0
        
        price = bs_barrier_premium(S, K, T, r, sigma, q, H, "call", "down-out")
        
        assert price >= 0
        
    def test_strike_equal_barrier_up_and_out_put(self):
        S = 100
        K = 105
        T = 1.0
        r = 0.05
        sigma = 0.2
        q = 0.0
        H = 105
        rebate = 0.0
        
        price = bs_barrier_premium(S, K, T, r, sigma, q, H, "put", "up-out")
        
        assert price >= 0
        
    def test_extreme_parameters(self):
        S = 100
        K = 100
        T = 0.1
        r = 0.5
        sigma = 1.0
        q = 0.1
        H = 95
        rebate = 0.0
        
        price = bs_barrier_premium(S, K, T, r, sigma, q, H, "call", "down-out")
        
        assert price >= 0
        assert not np.isnan(price)
        assert not np.isinf(price)
        
    def test_all_barrier_types_non_negative(self):
        S = 100
        K = 100
        T = 1.0
        r = 0.05
        sigma = 0.2
        q = 0.0
        H_down = 95
        H_up = 105
        rebate = 0.0
        
        barrier_types = ["down-out", "down-in", "up-out", "up-in"]
        option_types = ["call", "put"]
        
        for barrier_type in barrier_types:
            for option_type in option_types:
                H = H_down if "down" in barrier_type else H_up
                price = bs_barrier_premium(S, K, T, r, sigma, q, H, option_type, barrier_type)
                assert price >= 0, f"Negative price for {option_type} {barrier_type}"
                
            
    def test_moneyness_effect_calls(self):
        S = 100
        K_itm = 95
        K_atm = 100
        K_otm = 105
        T = 1.0
        r = 0.05
        sigma = 0.2
        q = 0.0
        H = 90
        rebate = 0.0
        
        price_itm = bs_barrier_premium(S, K_itm, T, r, sigma, q, H, "call", "down-out")
        price_atm = bs_barrier_premium(S, K_atm, T, r, sigma, q, H, "call", "down-out")
        price_otm = bs_barrier_premium(S, K_otm, T, r, sigma, q, H, "call", "down-out")
        
        assert price_itm >= price_atm >= price_otm
        
    def test_moneyness_effect_puts(self):
        S = 100
        K_itm = 105
        K_atm = 100
        K_otm = 95
        T = 1.0
        r = 0.05
        sigma = 0.2
        q = 0.0
        H = 110
        rebate = 0.0
        
        price_itm = bs_barrier_premium(S, K_itm, T, r, sigma, q, H, "put", "up-out")
        price_atm = bs_barrier_premium(S, K_atm, T, r, sigma, q, H, "put", "up-out")
        price_otm = bs_barrier_premium(S, K_otm, T, r, sigma, q, H, "put", "up-out")
        
        assert price_itm >= price_atm >= price_otm

if __name__ == "__main__":
    pytest.main([__file__])
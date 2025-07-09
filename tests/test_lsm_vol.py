import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.models.vanilla_options.american_options.longstaff_schwartz.pricing import lsm_american_premium
from core.models.vanilla_options.american_options.longstaff_schwartz.vol import AmericanImpliedVolSolver

def test_lsm_implied_vol():

    base_params = {
        'S0': 100,
        'T': 0.5,
        'r': 0.05,
        'q': 0.02,
        'option_type': 'put',
        'N': 30,
        'nb_paths': 100000,
        'seed': 42
    }
    
    strikes = [90, 95, 100, 105, 110]
    true_vol = 0.30

    for K in strikes:

        market_price = lsm_american_premium(
            base_params['S0'], K, base_params['T'], base_params['r'], true_vol,
            base_params['q'], base_params['N'], base_params['nb_paths'],
            base_params['option_type'], seed=base_params['seed']
        )[0]

        solver = AmericanImpliedVolSolver(
            K=K, market_price=market_price, **base_params
        )
        result = solver.solve_adaptive(tolerance=1e-4)
        print(result['implied_vol'])
        assert np.allclose(result['implied_vol'], true_vol, atol=1e-4)

if __name__ == "__main__":
    pytest.main(["tests/test_lsm_vol.py"])
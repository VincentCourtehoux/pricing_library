from pricing_library.core.pricing_service import PricingService
import numpy as np

service = PricingService()

def _ensure_list(x):
    if isinstance(x, (int, float, np.float64)):
        return [x]
    elif isinstance(x, (list, np.ndarray)):
        return list(x)
    else:
        return [x]

def print_greeks(title, greeks):
    print(f"\n{'='*50}\n{title}\n{'='*50}")

    values = greeks.get('greeks', greeks)

    price = _ensure_list(values.get('price', greeks.get('price', ['N/A'])))
    delta = _ensure_list(values.get('delta', ['N/A']*len(price)))
    gamma = _ensure_list(values.get('gamma', ['N/A']*len(price)))
    vega  = _ensure_list(values.get('vega', ['N/A']*len(price)))
    theta = _ensure_list(values.get('theta', ['N/A']*len(price)))
    rho   = _ensure_list(values.get('rho', ['N/A']*len(price)))

    print(f"{'Price':>10} {'Delta':>10} {'Gamma':>10} {'Vega':>10} {'Theta':>10} {'Rho':>10}")
    print("-"*70)
    for p, d, g, v, t, r in zip(price, delta, gamma, vega, theta, rho):
        p_str = f"{p:10.4f}" if isinstance(p, (int, float, np.float64)) else f"{p:>10}"
        d_str = f"{d:10.4f}" if isinstance(d, (int, float, np.float64)) else f"{d:>10}"
        g_str = f"{g:10.6f}" if isinstance(g, (int, float, np.float64)) else f"{g:>10}"
        v_str = f"{v:10.4f}" if isinstance(v, (int, float, np.float64)) else f"{v:>10}"
        t_str = f"{t:10.6f}" if isinstance(t, (int, float, np.float64)) else f"{t:>10}"
        r_str = f"{r:10.4f}" if isinstance(r, (int, float, np.float64)) else f"{r:>10}"

        print(f"{p_str} {d_str} {g_str} {v_str} {t_str} {r_str}")


# ===============================
# European Options - Black Scholes
# ===============================
params_european_bs = {
    'option_style': 'european',
    'S': [100, 105, 110],
    'K': [100, 100, 100],
    'T': [1, 0.5, 2],
    'r': [0.05, 0.05, 0.05],
    'q': [0, 0, 0],
    'sigma': [0.2, 0.25, 0.3],
    'option_type': ['call', 'put', 'call']
}

greeks_bs = service.calculate_greeks(params_european_bs, method='black_scholes')
print_greeks("European Options - Black Scholes", greeks_bs)

# ===============================
# European Options - Monte Carlo
# ===============================
params_european_mc = {
    'option_style': 'european',
    'S': 100,
    'K': 100,
    'T': 1,
    'r': 0.05,
    'sigma': 0.2,
    'option_type': 'call'
}

greeks_mc = service.calculate_greeks(params_european_mc, method='monte_carlo')
print_greeks("European Options - Monte Carlo", greeks_mc)

# ===============================
# European Options - Binomial Tree
# ===============================
params_european_bin = {
    'option_style': 'european',
    'S': 100,
    'K': 100,
    'T': 1,
    'r': 0.05,
    'sigma': 0.2,
    'option_type': 'call'
}

greeks_bin = service.calculate_greeks(params_european_bin, method='binomial_tree')
print_greeks("European Options - Binomial Tree", greeks_bin)

# ===============================
# American Options - Binomial Tree
# ===============================
params_american = {
    'option_style': 'american',
    'S': 100,
    'K': 100,
    'T': 1,
    'r': 0.05,
    'sigma': 0.2,
    'option_type': 'put'
}

price_am_bin = service.calculate_greeks(params_american, method='binomial_tree')
print_greeks("American Option - Binomial Tree", price_am_bin)

# ===============================
# American Options - LSM
# ===============================
params_american = {
    'option_style': 'american',
    'S': 100,
    'K': 100,
    'T': 1,
    'r': 0.05,
    'sigma': 0.2,
    'option_type': 'put'
}

price_am_lsm = service.calculate_greeks(params_american, method='least_squares_mc')
print_greeks("American Option - LSM", price_am_lsm)

# ===============================
# Asian Options - Monte Carlo
# ===============================
params_asian = {}
    #To Be Added

# ===============================
# Gap Options 
# ===============================
params_gap = {
    'option_style': 'gap',
    'S': [100, 110],
    'K1': [100, 105],
    'K2': [100, 105],
    'T': [1, 0.5],
    'r': [0.05, 0.05],
    'sigma': [0.2, 0.25],
    'q': [0, 0],
    'option_type': ['call', 'put']
}

greeks_gap = service.calculate_greeks(params_gap, method='black_scholes')
print_greeks("Gap Options", greeks_gap)

# ===============================
# Barrier Options
# ===============================
params_barrier = {
    'option_style': 'barrier',
    'S': 100,
    'K': 100,
    'T': 1,
    'r': 0.05,
    'sigma': 0.2,
    'option_type': 'call',
    'barrier_type': 'up-and-in',
    'barrier_level': 110
}

greeks_barrier = service.calculate_greeks(params_barrier, method='black_scholes')
print_greeks("Barrier Options", greeks_barrier)

# ===============================
# Combinations - Straddle 
# ===============================
params_straddle = {
    'option_style': 'combination',
    'combination': 'straddle',
    'S': [100, 105],
    'K': [100, 100],
    'T': [1, 0.5],
    'r': [0.05, 0.05],
    'q': [0, 0],
    'sigma': [0.2, 0.25]
}

greeks_straddle = service.calculate_greeks(params_straddle, method='black_scholes')
print_greeks("Straddle Combination", greeks_straddle)

# ===============================
# Combinations - Strangle 
# ===============================
params_strangle = {
    'option_style': 'combination',
    'combination': 'strangle',
    'S': [100, 105],
    'K_call': [105, 110],
    'K_put': [95, 100],
    'T': [1, 0.5],
    'r': [0.05, 0.05],
    'q': [0, 0],
    'sigma': [0.2, 0.25]
}

greeks_strangle = service.calculate_greeks(params_strangle, method='black_scholes')
print_greeks("Strangle Combination", greeks_strangle)

# ===============================
# Combinations - Bull Spread 
# ===============================
params_bull = {
    'option_style': 'combination',
    'combination': 'bull_spread',
    'S': [100, 105],
    'K1': [100, 100],
    'K2': [110, 110],
    'T': [1, 0.5],
    'r': [0.05, 0.05],
    'q': [0, 0],
    'sigma': [0.2, 0.25]
}

greeks_bull = service.calculate_greeks(params_bull, method='black_scholes')
print_greeks("Bull Spread Combination", greeks_bull)

# ===============================
# Combinations - Bear Spread
# ===============================
params_bear = {
    'option_style': 'combination',
    'combination': 'bear_spread',
    'S': [100, 105],
    'K1': [100, 100],
    'K2': [90, 95],
    'T': [1, 0.5],
    'r': [0.05, 0.05],
    'q': [0, 0],
    'sigma': [0.2, 0.25]
}

greeks_bear = service.calculate_greeks(params_bear, method='black_scholes')
print_greeks("Bear Spread Combination", greeks_bear)

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.models.vanilla_options.european_options.black_scholes.pricing_scalar import BlackScholesScalar

def _plot_single_greek(greek_name, compute_fn, S0, K, T, r, sigma, q, option_type, needs_option_type=False):
    moneyness = np.linspace(0.01 * K, 2 * K, 500)
    values = [
        compute_fn(s, K, T, r, sigma, q, option_type) if needs_option_type else compute_fn(s, K, T, r, sigma, q)
        for s in moneyness
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(moneyness * 100 / K, values, label=f"{option_type.capitalize()} {greek_name.capitalize()}", color="black")

    ax.set_xlabel("Moneyness (S / K)")
    tick_values = np.arange(0, 2.01, 0.25)
    ax.set_xticks(tick_values * 100)
    ax.set_xticklabels([f"{int(x * 100)}%" for x in tick_values])
    ax.set_ylabel(greek_name.capitalize())
    ax.set_title(f"{greek_name.capitalize()} of a {T}-year European {option_type.lower()} struck at {K / S0:.0%}", fontsize=10)
    ax.axvline(x=S0 * 100 / K, color='gray', linestyle='--', linewidth=0.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_greeks_european_options(S0, K, T, r, sigma, q, option_type):
    bs = BlackScholesScalar()

    delta_fig = _plot_single_greek("delta", bs.delta, S0, K, T, r, sigma, q, option_type, needs_option_type=True)
    gamma_fig = _plot_single_greek("gamma", bs.gamma, S0, K, T, r, sigma, q, option_type)
    vega_fig = _plot_single_greek("vega", bs.numerical_vega, S0, K, T, r, sigma, q, option_type)
    theta_fig = _plot_single_greek("theta", bs.numerical_daily_theta, S0, K, T, r, sigma, q, option_type, needs_option_type=True)
    rho_fig = _plot_single_greek("rho", bs.numerical_rho, S0, K, T, r, sigma, q, option_type, needs_option_type=True)

    return delta_fig, gamma_fig, vega_fig, theta_fig, rho_fig

delta_fig, gamma_fig, vega_fig, theta_fig, rho_fig = plot_greeks_european_options(100, 100, 1, 0.05, 0.2, 0.01, "call")
plt.show()


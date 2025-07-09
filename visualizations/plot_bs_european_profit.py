import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.models.vanilla_options.european_options.black_scholes.pricing_scalar import BlackScholesScalar

def plot_profit_european_options(S, K, T, r, sigma, q, option_type, position):

    S_plot = np.linspace(0, 2 * S, 500)
    S_percent = S_plot / S

    price = BlackScholesScalar().bs_european_scalar_premium(S, K, T, r, sigma, q, option_type)

    if option_type == "call":
        payoff = np.maximum(S_plot - K, 0)
        option_type_str = "Call"
    else:
        payoff = np.maximum(K - S_plot, 0)
        option_type_str = "Put"

    if position == "Long":
        profit = payoff - price
    else:
        profit = price - payoff

    price_percent = price / S
    payoff_percent = payoff / S
    profit_percent = profit / S

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(S_percent, profit_percent, color="black")
    ax.set_xticks(np.arange(0, 2.01, 0.25))
    ax.set_xticklabels([f"{int(x * 100)}%" for x in np.arange(0, 2.01, 0.25)])
    ax.set_xlim(0, 2)

    profit_min = np.floor(profit_percent.min() / 0.2) * 0.2
    profit_max = np.ceil(profit_percent.max() / 0.2) * 0.2
    yticks = np.arange(profit_min, profit_max + 0.01, 0.2)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{round(y * 100)}%" for y in yticks]) 

    ax.axhline(0, color='black', linewidth=0.8)
    ax.spines['bottom'].set_position(('data', 0))
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.yaxis.grid(True)
    ax.set_title(f"Profit from a {position.lower()} {option_type_str.lower()} position struck at {K / S:.0%}", fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig
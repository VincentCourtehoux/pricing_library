import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.models.vanilla_options.european_options.black_scholes.pricing_scalar import BlackScholesScalar

def plot_payoff_european_options(S, K, T, r, sigma, q, option_type):

    S_plot = np.linspace(0, 2 * S, 500)
    S_percent = S_plot / S

    price = BlackScholesScalar().bs_european_scalar_premium(S, K, T, r, sigma, q, option_type)

    if option_type == "call":
        payoff = np.maximum(S_plot - K, 0)
        option_type_str = "Call"
    else:
        payoff = np.maximum(K - S_plot, 0)
        option_type_str = "Put"

    price_percent = price / S
    payoff_percent = payoff / S

    price_plot = BlackScholesScalar().bs_european_scalar_premium(S_plot, K, T, r, sigma, q, option_type)
    price_percent_plot = price_plot / S

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(S_percent, price_percent_plot, label=f"{option_type_str} Price", color="black")
    ax.plot(S_percent, payoff_percent, label=f"{option_type_str} Payoff", linestyle="--", color="black")
    ax.set_xticks(np.arange(0, 2.01, 0.25))
    ax.set_xticklabels([f"{int(x * 100)}%" for x in np.arange(0, 2.01, 0.25)])
    ax.set_xlim(0, 2)

    yticks2 = np.arange(0, 1.01, 0.2)
    ax.set_yticks(yticks2)
    ax.set_yticklabels([f"{round(y * 100)}%" for y in yticks2]) 
    ax.set_ylim(0, 1)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.yaxis.grid(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    ax.set_title(f"Price of a {T}-year European {option_type.lower()} struck at {K / S:.0%}", fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    return fig
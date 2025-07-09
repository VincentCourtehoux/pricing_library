import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.models.vanilla_options.american_options.longstaff_schwartz.pricing import lsm_american_premium
from core.models.vanilla_options.european_options.black_scholes.pricing_scalar import BlackScholesScalar

def plot_lsm_convergence(S0, K, T, r, sigma, q, N, option_type, min_paths=100, max_paths=100000, num_points=20):    
    path_counts = np.unique(np.round(np.geomspace(min_paths, max_paths, num_points)).astype(int))
    
    bench_time = time.time()
    benchmark_price = lsm_american_premium(S0, K, T, r, sigma, q, N, max_paths, option_type, seed=42)[0]
    bench_time = time.time() - bench_time 
    european_price = BlackScholesScalar().bs_european_scalar_premium(S0, K, T, r, sigma, q, option_type) 
    
    prices = []
    std_errors = []
    computation_times = []

    for nb_paths in path_counts[:-1]:
        num_runs = 5
        run_prices = []
        
        start_time = time.time()
        for run in range(num_runs):
            price = lsm_american_premium(S0, K, T, r, sigma, q, N, nb_paths, option_type)[0]
            run_prices.append(price)
        end_time = time.time()
        
        avg_price = np.mean(run_prices)
        std_error = np.std(run_prices, ddof=1) / np.sqrt(num_runs)
        avg_time = (end_time - start_time) / num_runs
        
        prices.append(avg_price)
        std_errors.append(std_error)
        computation_times.append(avg_time)

    prices.append(benchmark_price)
    std_errors.append(0)
    computation_times.append(bench_time)
    
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.semilogx(path_counts, prices, linestyle='-', marker='o', color='blue', alpha=0.8, linewidth=2, markersize=6, label='LSM Price')
    ax1.axhline(y=benchmark_price, color='k', linestyle='--', linewidth=2, label=f'Benchmark {max_paths} Paths ({benchmark_price:.6f})')
    ax1.axhline(y=european_price, color='gray', linestyle=':', linewidth=2, label=f'European Price ({european_price:.6f})')
    ax1.fill_between(path_counts, 
                     np.array(prices) - 1.96 * np.array(std_errors), 
                     np.array(prices) + 1.96 * np.array(std_errors), 
                     alpha=0.6, color='lightgray', label='95% Confidence Interval')
    ax1.set_xlabel('Number of Monte Carlo Paths')
    ax1.set_ylabel('Option Price')
    ax1.set_title('Convergence of American Option Pricing (LSM)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.loglog(path_counts, computation_times, 'k-o', linewidth=2, markersize=6, label='Computation Time')
    ax2.set_xlabel('Number of Monte Carlo Paths')
    ax2.set_ylabel('Computation Time (seconds)')
    ax2.set_title('Computation Time vs. Number of Paths')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig1, fig2
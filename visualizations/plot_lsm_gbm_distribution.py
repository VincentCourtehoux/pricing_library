import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.models.vanilla_options.american_options.longstaff_schwartz.pricing import lsm_american_premium

def plot_exercise_nodes_american_options_with_distribution(S0, K, T, r, sigma, q, N, nb_paths_to_plot=100, total_paths=10000, figsize=(14, 8)):
    """
    Plot LSM simulation paths showing exercise decisions with final price distribution
    
    Parameters:
    S0 (float): Initial asset price
    K (float): Strike price
    T (float): Time to maturity
    r (float): Risk-free rate
    sigma (float): Volatility
    q (float): Dividend yield
    N (int): Number of time steps
    nb_paths_to_plot (int): Number of paths to display
    total_paths (int): Total number of paths for the full simulation (for distribution)
    figsize (tuple): Figure size

    Returns:
    fig (matplotlib.figure.Figure): The matplotlib figure object containing the plot.
    """
    premium_display, S_paths_display, exercise_matrix_display, time_grid = lsm_american_premium(
        S0, K, T, r, sigma, q, N, nb_paths_to_plot, option_type='put', degree=2, seed=42
    )

    premium_full, S_paths_full, exercise_matrix_full, _ = lsm_american_premium(
        S0, K, T, r, sigma, q, N, total_paths, option_type='put', degree=2, seed=123
    )
    
    fig, ax = plt.subplots(figsize=figsize)
    
    nb_paths = min(nb_paths_to_plot, S_paths_display.shape[0])
    selected_paths = np.random.choice(S_paths_display.shape[0], nb_paths, replace=False)
    
    for i, path_idx in enumerate(selected_paths):
        path = S_paths_display[path_idx, :]
        exercise_decisions = exercise_matrix_display[path_idx, :]
   
        exercise_times = np.where(exercise_decisions)[0]
        if len(exercise_times) > 0:
            first_exercise_time = exercise_times[0]
  
            ax.plot(time_grid[:first_exercise_time+1], path[:first_exercise_time+1], 
                   color='lightblue', alpha=1, linewidth=1)

            ax.plot(time_grid[first_exercise_time], path[first_exercise_time], 
                   'ro', markersize=3, alpha=0.7)

            if first_exercise_time < len(time_grid) - 1:
                ax.plot(time_grid[first_exercise_time:], path[first_exercise_time:], 
                       color='lightgray', linestyle='--', alpha=0.5, linewidth=1)
        else:
            ax.plot(time_grid, path, color='lightblue', alpha=1, linewidth=1)

    ax.axhline(y=K, color='red', linestyle='--', alpha=0.8, linewidth=1, label=f'Strike K = {K}')

    final_prices = S_paths_full[:, -1]

    n_bins = 50
    counts, bins = np.histogram(final_prices, bins=n_bins, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]

    max_count = np.max(counts)
    time_range = time_grid[-1] - time_grid[0]
    scale_factor = 0.15 * time_range / max_count

    for i in range(len(counts)):
        if counts[i] > 0:
            ax.barh(bin_centers[i], counts[i] * scale_factor, 
                   height=bin_width, left=time_grid[-1] + 0.02, 
                   alpha=0.7, color='lightcoral', edgecolor='black', linewidth=0.3)
    
    ax.set_xlabel('Time t', fontsize=12)
    ax.set_ylabel('Underlying Asset Price X(t)', fontsize=12)
    ax.set_title(f'Simulated Paths with Exercise Decisions and Final Price Distribution\n(Distribution based on {total_paths:,} paths)', fontsize=14)
    ax.grid(True, alpha=0.3)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='lightblue', alpha=1, label='Path before exercise'),
        Line2D([0], [0], color='lightgray', linestyle='--', alpha=0.5, label='Path after exercise'),
        Line2D([0], [0], color='red', marker='o', linestyle='None', 
               markersize=5, label='First favourable exercise'),
        Line2D([0], [0], color='red', linestyle='--', alpha=0.8, label='Strike K'),
        Line2D([0], [0], color='lightcoral', alpha=0.7, label=f'Final price distribution ({total_paths:,} paths)')
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    plt.tight_layout()
    return fig
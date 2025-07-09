import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.models.vanilla_options.american_options.longstaff_schwartz.pricing import lsm_american_premium

def plot_exercise_nodes_american_options(S0, K, T, r, sigma, q, N, nb_paths_to_plot=100, figsize=(12, 8)):
    """
    Plot LSM simulation paths showing exercise decisions
    
    Parameters:
    S_paths: Asset price paths
    exercise_matrix: Boolean matrix indicating exercise decisions
    time_grid: Time points
    K: Strike price
    nb_paths_to_plot: Number of paths to display
    figsize: Figure size
    """
    premium, S_paths, exercise_matrix, time_grid = lsm_american_premium(
        S0, K, T, r, sigma, q, N, nb_paths_to_plot, option_type='put', degree=2, seed=42
    )
    fig, ax = plt.subplots(figsize=figsize)
    
    nb_paths = min(nb_paths_to_plot, S_paths.shape[0])
    selected_paths = np.random.choice(S_paths.shape[0], nb_paths, replace=False)
    
    for i, path_idx in enumerate(selected_paths):
        path = S_paths[path_idx, :]
        exercise_decisions = exercise_matrix[path_idx, :]
   
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
 
    ax.axhline(y=K, color='black', linestyle='--', alpha=0.8, linewidth=0.5, label=f'Strike K = {K}')
    
    ax.set_xlabel('Time t', fontsize=12)
    ax.set_ylabel('Underlying Asset Price X(t)', fontsize=12)
    ax.set_title('Simulated Paths\nshowing Exercise or Continuation Decision', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='lightblue', alpha=1, label='Path before exercise'),
        Line2D([0], [0], color='lightgray', linestyle='--', alpha=0.5, label='Path after exercise'),
        Line2D([0], [0], color='red', marker='o', linestyle='None', 
               markersize=5, label='First favourable exercise'),
        Line2D([0], [0], color='black', linestyle='--', alpha=0.8, label='Strike K')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    return fig
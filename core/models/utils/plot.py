import matplotlib.pyplot as plt

def plot_paths(paths, n_paths_to_plot=10, title="Simulated Asset Price Paths"):
    plt.figure(figsize=(10, 5))
    for i in range(min(n_paths_to_plot, paths.shape[0])):
        plt.plot(paths[i], lw=1)
    plt.xlabel("Time Steps")
    plt.ylabel("Asset Price")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
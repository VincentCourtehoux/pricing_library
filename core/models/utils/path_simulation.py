import numpy as np

def simulate_gbm_paths(S0, T, r, sigma, q, N, nb_paths):
    dt = T / N
    paths = np.zeros((nb_paths, N + 1))
    paths[:, 0] = S0

    for t in range(1, N + 1):
        Z = np.random.standard_normal(nb_paths)
        paths[:, t] = paths[:, t - 1] * np.exp((r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

    return paths
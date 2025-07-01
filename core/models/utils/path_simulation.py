import numpy as np

def simulate_gbm(S0, r, sigma, T, q, N, nb_paths, seed=None):
    dt = T / N
    rng = np.random.default_rng(seed)
    S = np.zeros((N+1, nb_paths))
    S[0] = S0
    for t in range(1, N+1):
        z = rng.standard_normal(nb_paths)
        S[t] = S[t-1] * np.exp((r - q - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
    return S
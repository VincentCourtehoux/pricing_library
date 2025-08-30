import numpy as np

class GeometricBrownianMotion:
    
    @staticmethod
    def simulate(S0, T, r, sigma, q=0.0, n_paths=10000, n_steps=100, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        dt = T / n_steps
        z = np.random.normal(0, 1, (n_paths, n_steps))
        
        increments = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        log_returns = np.cumsum(increments, axis=1)

        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = S0
        paths[:, 1:] = S0 * np.exp(log_returns)
        
        return paths
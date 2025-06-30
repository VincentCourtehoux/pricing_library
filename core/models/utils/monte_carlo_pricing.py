import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

class MonteCarloPricing:

    def simulate_paths(self, S0, T, r, sigma, q, N, nb_paths):
        """
        Simulates asset price paths using the geometric Brownian motion model.

        Parameters:
        S0 (float): Initial asset price
        T (float): Time to maturity (in years)
        r (float): Risk-free interest rate
        sigma (float): Volatility of the asset
        q (float): Continuous dividend yield
        N (int): Number of time steps
        nb_paths (int): Number of simulated paths

        Returns:
        np.ndarray: Simulated paths of shape (nb_sim, N+1)
        """
        dt = T / N
        paths = np.zeros((nb_paths, N + 1))
        paths[:, 0] = S0

        for t in range(1, N + 1):
            Z = np.random.standard_normal(nb_paths)
            paths[:, t] = paths[:, t - 1] * np.exp((r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

        return paths

    def compute_payoff(self, ST, K, option_type = "call"):
        """
        Computes the payoff of European call or put options.

        Parameters:
        ST (np.ndarray): Terminal asset prices
        K (float): Strike price
        option_type (str): 'call' or 'put'

        Returns:
        np.ndarray: Payoffs for each simulation
        """
        if option_type == "call":
            return np.maximum(ST - K, 0)
        elif option_type == "put":
            return np.maximum(K - ST, 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

    def confidence_interval(self, data, alpha = 0.05):
        """
        Computes the (1 - alpha)% confidence interval of the mean of the data.

        Parameters:
        data (np.ndarray): Array of sample values
        alpha (float): Significance level (default 0.05 for 95% confidence)

        Returns:
        tuple: (mean, margin of error)
        """
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        n = len(data)
        z = norm.ppf(1 - alpha / 2)
        margin = z * std / np.sqrt(n)
        return mean, margin

    def monte_carlo_pricing(self, S0, K, T, r, sigma, q, N, nb_paths, option_type="call", return_all=False):
        """
        Estimates the price of a European option using Monte Carlo simulation.

        Parameters:
        S0 (float): Initial asset price
        K (float): Strike price
        T (float): Time to maturity (in years)
        r (float): Risk-free interest rate
        sigma (float): Volatility of the asset
        q (float): Continuous dividend yield
        N (int): Number of time steps
        nb_paths (int): Number of simulated paths
        option_type (str): 'call' or 'put'
        return_all (bool): If True, returns detailed results

        Returns:
        float: Option price estimate (if return_all=False)
        dict: Contains price, confidence interval, paths, payoffs, and discounted payoffs (if return_all=True)
        """
        paths = self.simulate_paths(S0, T, r, sigma, q, N, nb_paths)
        ST = paths[:, -1]
        payoff = self.compute_payoff(ST, K, option_type)
        discounted = np.exp(-r * T) * payoff
        mean, margin = self.confidence_interval(discounted)

        if return_all:
            return {
                "price": mean,
                "confidence_interval": (mean - margin, mean + margin),
                "paths": paths,
                "payoffs": payoff,
                "discounted_payoffs": discounted
            }
        else:
            return mean
        
    def plot_paths(self, paths, N):
        """
        Plot a subset of simulated asset price paths.

        Parameters:
        paths (np.ndarray): Simulated paths of shape (nb_sim, N+1)
        n_paths_to_plot (int): Number of paths to plot
        title (str): Plot title
        """
        plt.figure(figsize=(10, 5))
        for i in range(min(N, paths.shape[0])):
            plt.plot(paths[i], lw=1)
        plt.xlabel("Time Steps")
        plt.ylabel("Asset Price")
        plt.title("Simulated Asset Price Paths")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    

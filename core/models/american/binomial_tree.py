import numpy as np
import sys
import os
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.payoff import compute_payoff

class BinomialTreeAmerican:
    def __init__(self, S0, K, T, r, sigma, n, option_type='call'):
        """
        Binomial Tree model for American option pricing.

        Parameters:
        S0 (float): Initial stock price.
        K (float): Strike price of the option.
        T (float): Time to maturity (in years).
        r (float): Risk-free interest rate.
        sigma (float): Volatility of the underlying asset.
        n (int): Number of time steps in the binomial tree.
        option_type (str): Type of option, either 'call' or 'put'. Default is 'call'.
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.n = n
        self.option_type = option_type.lower()

        self.dt = T / n
        self.u = np.exp(sigma * np.sqrt(self.dt))
        self.d = 1 / self.u
        self.p = (np.exp(r * self.dt) - self.d) / (self.u - self.d)
        self.discount_factor = np.exp(-r * self.dt)

    def build_stock_tree(self):
        """
        Build the stock price tree.

        Returns:
        stock_tree (np.ndarray): A (n+1)x(n+1) matrix representing stock prices at each node.
        """
        stock_tree = np.zeros((self.n + 1, self.n + 1))

        for i in range(self.n + 1):
            for j in range(i + 1):
                stock_tree[j, i] = self.S0 * (self.u ** (i - j)) * (self.d ** j)

        return stock_tree

    def price_option(self):
        """
        Compute the American option price using backward induction.

        Returns:
        option_price (float): The price of the American option.
        """
        stock_tree = self.build_stock_tree()
        option_tree = np.zeros((self.n + 1, self.n + 1))

        for j in range(self.n + 1):
            option_tree[j, self.n] = compute_payoff(stock_tree[j, self.n], self.K, self.option_type)

        for i in range(self.n - 1, -1, -1):
            for j in range(i + 1):
                continuation_value = self.discount_factor * (
                    self.p * option_tree[j, i + 1] +
                    (1 - self.p) * option_tree[j + 1, i + 1]
                )

                exercise_value = compute_payoff(stock_tree[j, i], self.K, self.option_type)
                option_tree[j, i] = max(continuation_value, exercise_value)

        self.stock_tree = stock_tree
        self.option_tree = option_tree

        return option_tree[0, 0]

    def get_exercise_boundary(self):
        """
        Identify early exercise boundary (nodes where exercising is optimal).

        Returns:
        exercise_nodes (list of tuple): Each tuple is (i, j, stock_price), where i is the time step,
                                        j is the node index, and stock_price is the asset price.
        """
        exercise_nodes = []

        for i in range(self.n):
            for j in range(i + 1):
                continuation_value = self.discount_factor * (
                    self.p * self.option_tree[j, i + 1] +
                    (1 - self.p) * self.option_tree[j + 1, i + 1]
                )
                exercise_value = compute_payoff(self.stock_tree[j, i], self.K, self.option_type)

                if abs(self.option_tree[j, i] - exercise_value) < 1e-10 and exercise_value > 0:
                    exercise_nodes.append((i, j, self.stock_tree[j, i]))

        return exercise_nodes
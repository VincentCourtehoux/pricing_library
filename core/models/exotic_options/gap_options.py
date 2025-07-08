import numpy as np
from scipy.stats import norm

class BlackScholesGap:

    def validation_option_type(self, option_type):
        """
        Validate the option type string and normalize it to 'call' or 'put'.

        Parameters:
        option_type (str): Input option type string, expected to start with 'c' or 'p'

        Returns:
        str: 'call' or 'put' normalized option type

        Raises:
        ValueError: If the option type does not start with 'c' or 'p'
        """
        option_type = option_type.lower()
        if option_type.startswith("c"):
            return "call"
        elif option_type.startswith("p"):
            return "put"
        else:
            raise ValueError("Option type must start with 'c' or 'p' (e.g. 'call', 'put')")

    def d1(self, S, K2, T, r, sigma, q=0.0):
        """
        Calculate the d1 component of the Black-Scholes formula.

        Parameters:
        S (float): Current price of the underlying asset
        K2 (float): Trigger price
        T (float): Time to maturity (in years)
        r (float): Risk-free interest rate (annual)
        sigma (float): Volatility of the underlying asset (annual)
        q (float): Continuous dividend yield

        Returns:
        float: The calculated d1 value
        """
        if T > 0:
            return (np.log(S / K2) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        else:
            return np.nan

    def d2(self, S, K2, T, r, sigma, q):
        """
        Calculate the d2 component of the Black-Scholes formula.

        Parameters:
        S (float): Current price of the underlying asset
        K2 (float): Trigger price
        T (float): Time to maturity (in years)
        r (float): Risk-free interest rate (annual)
        sigma (float): Volatility of the underlying asset (annual)

        Returns:
        float: The calculated d2 value
        """
        return self.d1(S, K2, T, r, sigma, q) - sigma * np.sqrt(T)

    def bs_gap_premium(self, S, K1, K2, T, r, sigma, q=0.0, option_type="call"):
        """
        Compute the Black-Scholes price for a call or put gap option.

        Parameters:
        S (float): Current price of the underlying asset
        K1 (float): Strike price
        K2 (float): Trigger price
        T (float): Time to maturity (in years)
        r (float): Risk-free interest rate (annual)
        sigma (float): Volatility of the underlying asset (annual)
        q (float): Continuous dividend yield
        option_type (str): 'call' or 'put'

        Returns:
        float: Theoretical option premium (price)
        """
        option_type = self.validation_option_type(option_type)

        if T == 0 and S > K2 and option_type == "call":
            return max(S - K1, 0) 
        elif T == 0 and S <= K2 and option_type == "call":
            return 0
        elif T == 0 and S < K2 and option_type == "put":
            return max(K1 - S, 0)
        elif T == 0 and S >= K2 and option_type == "put":
            return 0

        if sigma == 0 and S > K2 and option_type == "call":
            return max(S * np.exp(-q * T) - K1 * np.exp(-r * T), 0)
        elif sigma == 0 and S <= K2 and option_type == "call":
            return 0
        elif sigma == 0 and S < K2 and option_type == "put":
            return max(K1 * np.exp(-r * T) - S * np.exp(-q * T), 0)
        elif sigma == 0 and S >= K2 and option_type == "put":
            return 0

        d1 = self.d1(S, K2, T, r, sigma, q)
        d2 = self.d2(S, K2, T, r, sigma, q)

        if option_type == "call":
            return max(S * np.exp(-q * T) * norm.cdf(d1) - K1 * np.exp(-r * T) * norm.cdf(d2), 0)
        else: 
            return max(K1 * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1), 0)




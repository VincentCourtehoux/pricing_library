import numpy as np
from scipy.stats import norm

class BlackScholesScalar:

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

    def d1(self, S, K, T, r, sigma, q=0.0):
        """
        Calculate the d1 component of the Black-Scholes formula.

        Parameters:
        S (float): Current price of the underlying asset
        K (float): Strike price
        T (float): Time to maturity (in years)
        r (float): Risk-free interest rate (annual)
        sigma (float): Volatility of the underlying asset (annual)
        q (float): Continuous dividend yield

        Returns:
        float: The calculated d1 value
        """
        if T > 0:
            return (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        else:
            return np.nan

    def d2(self, S, K, T, r, sigma, q):
        """
        Calculate the d2 component of the Black-Scholes formula.

        Parameters:
        S (float): Current price of the underlying asset
        K (float): Strike price
        T (float): Time to maturity (in years)
        r (float): Risk-free interest rate (annual)
        sigma (float): Volatility of the underlying asset (annual)

        Returns:
        float: The calculated d2 value
        """
        return self.d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)

    def bs_european_scalar_premium(self, S, K, T, r, sigma, q=0.0, option_type="call"):
        """
        Compute the Black-Scholes price for a European call or put option.

        Parameters:
        S (float): Current price of the underlying asset
        K (float): Strike price
        T (float): Time to maturity (in years)
        r (float): Risk-free interest rate (annual)
        sigma (float): Volatility of the underlying asset (annual)
        q (float): Continuous dividend yield
        option_type (str): 'call' or 'put'

        Returns:
        float: Theoretical option premium (price)
        """
        option_type = self.validation_option_type(option_type)

        if T == 0:
            return max(S - K, 0) if option_type == "call" else max(K - S, 0)

        if sigma == 0:
            return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0) if option_type == "call" else max(K * np.exp(-r * T) - S * np.exp(-q * T), 0)

        d1 = self.d1(S, K, T, r, sigma, q)
        d2 = self.d2(S, K, T, r, sigma, q)

        if option_type == "call":
            return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else: 
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    def delta(self, S, K, T, r, sigma, q=0.0, option_type="call"):
        """
        Calculate the delta of a European call or put option.

        Parameters:
        S (float): Current price of the underlying asset
        K (float): Strike price
        T (float): Time to maturity (in years)
        r (float): Risk-free interest rate (annual)
        sigma (float): Volatility of the underlying asset (annual)
        q (float): Continuous dividend yield
        option_type (str): 'call' or 'put'

        Returns:
        float: Delta value (sensitivity of option price to underlying price)
        """
        option_type = self.validation_option_type(option_type)
        d1 = self.d1(S, K, T, r, sigma, q)
        return np.exp(-q * T) * norm.cdf(d1) if option_type == "call" else np.exp(-q * T) * (norm.cdf(d1) - 1)

    def gamma(self, S, K, T, r, sigma, q=0.0):
        """
        Calculate the gamma of a European option (same for call and put).

        Parameters:
        S (float): Current price of the underlying asset
        K (float): Strike price
        T (float): Time to maturity (in years)
        r (float): Risk-free interest rate (annual)
        sigma (float): Volatility of the underlying asset (annual)
        q (float): Continuous dividend yield

        Returns:
        float: Gamma value (rate of change of delta with respect to underlying price)
        """
        d1 = self.d1(S, K, T, r, sigma, q)
        return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))

    def vega_analytical(self, S, K, T, r, sigma, q=0.0):
        """
        Calculate the vega of a European option (same for call and put).

        Parameters:
        S (float): Current price of the underlying asset
        K (float): Strike price
        T (float): Time to maturity (in years)
        r (float): Risk-free interest rate (annual)
        sigma (float): Volatility of the underlying asset (annual)
        q (float): Continuous dividend yield

        Returns:
        float: Vega value (sensitivity of option price to volatility)
        """
        d1 = self.d1(S, K, T, r, sigma, q)
        return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
    
    def numerical_vega(self, S, K, T, r, sigma, q=0.0, option_type="call", ds = 0.005):
        """
        Calculate the vega (volatility sensitivity) of a European call or put option.

        Parameters:
        S (float): Current price of the underlying asset
        K (float): Strike price
        T (float): Time to maturity (in years)
        r (float): Risk-free interest rate (annual)
        sigma (float): Volatility of the underlying asset (annual)
        q (float): Continuous dividend yield
        option_type (str): 'call' or 'put'
        ds (float): Step size for volatility (e.g., 0.01 = 1%)

        Returns:
        float: Vega value (sensitivity of option price to volatility changes)
        """
        return self.bs_european_scalar_premium(S, K, T, r, sigma + ds, q, option_type) - self.bs_european_scalar_premium(S, K, T, r, sigma - ds, q, option_type)

    def theta_analytical(self, S, K, T, r, sigma, q=0.0, option_type="call"):
        """
        Calculate the theta (time decay) of a European call or put option.

        Parameters:
        S (float): Current price of the underlying asset
        K (float): Strike price
        T (float): Time to maturity (in years)
        r (float): Risk-free interest rate (annual)
        sigma (float): Volatility of the underlying asset (annual)
        q (float): Continuous dividend yield
        option_type (str): 'call' or 'put'

        Returns:
        float: Theta value (rate of decline of option price over time)
        """
        option_type = self.validation_option_type(option_type)
        d1 = self.d1(S, K, T, r, sigma, q)
        d2 = self.d2(S, K, T, r, sigma, q)
        term1 = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))

        if option_type == "call":
            return term1 + q * S * np.exp(-q * T) * norm.cdf(d1) - r * K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return term1 - q * S * np.exp(-q * T) * norm.cdf(-d1) + r * K * np.exp(-r * T) * norm.cdf(-d2)
    
    def numerical_daily_theta(self, S, K, T, r, sigma, q=0, option_type="call", dt=1/252):
        """
        Compute the numerical (daily) theta of a European option.

        Parameters:
        S (float): Spot price
        K (float): Strike price
        T (float): Time to maturity (in years)
        r (float): Risk-free interest rate (annualized)
        sigma (float): Volatility (annualized)
        q (float): Continuous dividend yield
        option_type (str): 'call' or 'put'
        dt (float): Time step (in years), default is 1 day

        Returns:
        float: Approximate theta per day (negative means time decay)
        """
        return self.bs_european_scalar_premium(S, K, T - dt, r, sigma, q, option_type) - self.bs_european_scalar_premium(S, K, T, r, sigma, q, option_type)

    def rho_analytical(self, S, K, T, r, sigma, q=0.0, option_type="call"):
        """
        Calculate the rho (interest rate sensitivity) of a European call or put option.

        Parameters:
        S (float): Current price of the underlying asset
        K (float): Strike price
        T (float): Time to maturity (in years)
        r (float): Risk-free interest rate (annual)
        sigma (float): Volatility of the underlying asset (annual)
        q (float): Continuous dividend yield
        option_type (str): 'call' or 'put'

        Returns:
        float: Rho value (sensitivity of option price to interest rate changes)
        """
        option_type = self.validation_option_type(option_type)
        d2 = self.d2(S, K, T, r, sigma, q)

        if option_type == "call":
            return K * T * np.exp(-r * T) * norm.cdf(d2)
        else:
            return -K * T * np.exp(-r * T) * norm.cdf(-d2)
        
    def numerical_rho(self, S, K, T, r, sigma, q=0.0, option_type="call", dr=0.005):
        """
        Calculate the rho (interest rate sensitivity) of a European call or put option.

        Parameters:
        S (float): Current price of the underlying asset
        K (float): Strike price
        T (float): Time to maturity (in years)
        r (float): Risk-free interest rate (annual)
        sigma (float): Volatility of the underlying asset (annual)
        q (float): Continuous dividend yield
        option_type (str): 'call' or 'put'
        dr (float): Step size for interest rate (e.g., 0.01 = 1%)

        Returns:
        float: Rho value (sensitivity of option price to interest rate changes)
        """
        return self.bs_european_scalar_premium(S, K, T, r + dr, sigma, q, option_type) - self.bs_european_scalar_premium(S, K, T, r - dr, sigma, q, option_type)
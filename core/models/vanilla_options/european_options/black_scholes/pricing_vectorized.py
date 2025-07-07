import numpy as np
from scipy.stats import norm

class BlackScholesVectorized:

    def validation_option_type(self, option_type):
        """
        Validate and normalize option types for vector inputs.

        Parameters:
        option_type (array-like of str): Option types as strings, expected to start with 'c' or 'p'

        Returns:
        np.ndarray: Array of normalized option types 'call' or 'put'

        Raises:
        ValueError: If any option type does not start with 'c' or 'p'
        """
        option_type = np.char.lower(option_type)
        mask_call = np.char.startswith(option_type, "c")
        mask_put  = np.char.startswith(option_type, "p")

        if not np.all(mask_call | mask_put):
            raise ValueError("All option types must start with 'c' or 'p' (e.g. 'call', 'put')")

        result = np.empty(option_type.shape, dtype='<U4')  
        result[mask_call] = "call"
        result[mask_put]  = "put"
        return result

    def d1(self, S, K, T, r, sigma, q=0.0):
        """
        Vectorized calculation of d1 for Black-Scholes.

        Parameters:
        S (float or array-like): Current price(s) of the underlying asset
        K (float or array-like): Strike price(s)
        T (float or array-like): Time(s) to maturity (in years)
        r (float or array-like): Risk-free interest rate(s) (annual)
        sigma (float or array-like): Volatility(ies) of the underlying asset (annual)
        q (float or array-like): Continuous dividend yield(s)
        
        Returns:
        np.ndarray: d1 values
        """
        S     = np.asarray(S)
        K     = np.asarray(K)
        T     = np.asarray(T)
        r     = np.asarray(r)
        sigma = np.asarray(sigma)
        q     = np.asarray(q)

        d1    = np.full_like(T, np.nan, dtype=np.float64)
        mask  = T > 0

        d1[mask] = (
        np.log(S[mask] / K[mask])
        + (r[mask] - q[mask] + 0.5 * sigma[mask]**2) * T[mask]
        ) / (sigma[mask] * np.sqrt(T[mask]))

        d1[~mask] = np.nan
        return d1

    def d2(self, S, K, T, r, sigma, q=0.0):
        """
        Vectorized calculation of d2 for Black-Scholes.

        Parameters:
        S (float or array-like): Current price(s) of the underlying asset
        K (float or array-like): Strike price(s)
        T (float or array-like): Time(s) to maturity (in years)
        r (float or array-like): Risk-free interest rate(s) (annual)
        sigma (float or array-like): Volatility(ies) of the underlying asset (annual)
        q (float or array-like): Continuous dividend yield(s)

        Returns:
        np.ndarray: d2 values
        """
        S     = np.asarray(S)
        K     = np.asarray(K)
        T     = np.asarray(T)
        r     = np.asarray(r)
        sigma = np.asarray(sigma)
        q     = np.asarray(q)

        d2    = np.full_like(T, np.nan, dtype=np.float64)
        mask  = T > 0

        d1_vals = self.d1(S, K, T, r, sigma, q)
        d2[mask] = d1_vals[mask] - sigma[mask] * np.sqrt(T[mask])
        d2[~mask] = 0
        return d2

    def bs_european_vectorized_premium(self, S, K, T, r, sigma, q=0.0, option_type="call"):
        """
        Vectorized Black-Scholes price for European call or put options.

        Parameters:
        S (float or array-like): Current price(s) of the underlying asset
        K (float or array-like): Strike price(s)
        T (float or array-like): Time(s) to maturity (in years)
        r (float or array-like): Risk-free interest rate(s) (annual)
        sigma (float or array-like): Volatility(ies) of the underlying asset (annual)
        q (float or array-like): Continuous dividend yield(s)
        option_type (array-like of str): Option type(s) as string(s), expected to start with 'c' or 'p'

        Returns:
        np.ndarray: Option premiums (prices)
        """
        option_type = self.validation_option_type(option_type)

        S     = np.asarray(S)
        K     = np.asarray(K)
        T     = np.asarray(T)
        r     = np.asarray(r)
        sigma = np.asarray(sigma)
        q     = np.asarray(q)

        price = np.zeros_like(S, dtype=float)

        mask_T0      = (T == 0)
        mask_call_T0 = mask_T0 & (option_type == "call")
        mask_put_T0  = mask_T0 & (option_type == "put")
        price[mask_call_T0] = np.maximum(S[mask_call_T0] - K[mask_call_T0], 0)
        price[mask_put_T0]  = np.maximum(K[mask_put_T0] - S[mask_put_T0], 0)

        mask_s0       = (sigma == 0) & (T > 0)
        mask_call_s0  = mask_s0 & (option_type == "call")
        mask_put_s0   = mask_s0 & (option_type == "put")
        if np.any(mask_s0):
            price[mask_call_s0] = np.maximum(
                S[mask_call_s0] * np.exp(-q[mask_call_s0] * T[mask_call_s0]) -
                K[mask_call_s0] * np.exp(-r[mask_call_s0] * T[mask_call_s0]), 0
            )
            price[mask_put_s0] = np.maximum(
                K[mask_put_s0] * np.exp(-r[mask_put_s0] * T[mask_put_s0]) -
                S[mask_put_s0] * np.exp(-q[mask_put_s0] * T[mask_put_s0]), 0
            )

        mask_active = (T > 0) & (sigma > 0)
        if np.any(mask_active):
            idx = np.where(mask_active)[0]
            d1_vals = self.d1(S[idx], K[idx], T[idx], r[idx], sigma[idx], q[idx])
            d2_vals = self.d2(S[idx], K[idx], T[idx], r[idx], sigma[idx], q[idx])
            opt = option_type[mask_active]

            call_mask = opt == "call"
            put_mask = opt == "put"
            call_idx = idx[call_mask]
            put_idx = idx[put_mask]

            price[call_idx] = (
            S[call_idx] * np.exp(-q[call_idx] * T[call_idx]) * norm.cdf(d1_vals[call_idx])
            - K[call_idx] * np.exp(-r[call_idx] * T[call_idx]) * norm.cdf(d2_vals[call_idx])
            )

            price[put_idx] = (
            K[put_idx] * np.exp(-r[put_idx] * T[put_idx]) * norm.cdf(-d2_vals[put_idx])
            - S[put_idx] * np.exp(-q[put_idx] * T[put_idx]) * norm.cdf(-d1_vals[put_idx])
            )

        return price


    def delta(self, S, K, T, r, sigma, q=0.0, option_type="call"):
        """
        Vectorized calculation of delta for call and put options.

        Parameters:
        S (float or array-like): Current price(s) of the underlying asset
        K (float or array-like): Strike price(s)
        T (float or array-like): Time(s) to maturity (in years)
        r (float or array-like): Risk-free interest rate(s) (annual)
        sigma (float or array-like): Volatility(ies) of the underlying asset (annual)
        q (float or array-like): Continuous dividend yield(s)
        option_type (array-like of str): Option type(s) as string(s), expected to start with 'c' or 'p'

        Returns:
        np.ndarray: Delta values
        """
        option_type = self.validation_option_type(option_type)

        S     = np.asarray(S)
        K     = np.asarray(K)
        T     = np.asarray(T)
        r     = np.asarray(r)
        sigma = np.asarray(sigma)
        q     = np.asarray(q)

        d1_vals = self.d1(S, K, T, r, sigma, q)
        return np.where(option_type == "call", np.exp(-q * T) * norm.cdf(d1_vals), np.exp(-q * T) * (norm.cdf(d1_vals) - 1))

    def gamma(self, S, K, T, r, sigma, q=0.0):
        """
        Vectorized calculation of gamma for options.

        Parameters:
        S (float or array-like): Current price(s) of the underlying asset
        K (float or array-like): Strike price(s)
        T (float or array-like): Time(s) to maturity (in years)
        r (float or array-like): Risk-free interest rate(s) (annual)
        sigma (float or array-like): Volatility(ies) of the underlying asset (annual)
        q (float or array-like): Continuous dividend yield(s)

        Returns:
        np.ndarray: Gamma values
        """
        S     = np.asarray(S)
        K     = np.asarray(K)
        T     = np.asarray(T)
        r     = np.asarray(r)
        sigma = np.asarray(sigma)
        q     = np.asarray(q)

        d1_vals = self.d1(S, K, T, r, sigma, q)
        return np.exp(-q * T) * norm.pdf(d1_vals) / (S * sigma * np.sqrt(T))

    def vega(self, S, K, T, r, sigma, q=0.0):
        """
        Vectorized calculation of vega for options.

        Parameters:
        S (float or array-like): Current price(s) of the underlying asset
        K (float or array-like): Strike price(s)
        T (float or array-like): Time(s) to maturity (in years)
        r (float or array-like): Risk-free interest rate(s) (annual)
        sigma (float or array-like): Volatility(ies) of the underlying asset (annual)
        q (float or array-like): Continuous dividend yields

        Returns:
        np.ndarray: Vega values
        """
        S     = np.asarray(S)
        K     = np.asarray(K)
        T     = np.asarray(T)
        r     = np.asarray(r)
        sigma = np.asarray(sigma)
        q     = np.asarray(q)

        d1_vals = self.d1(S, K, T, r, sigma, q)
        return S * np.exp(-q * T) * norm.pdf(d1_vals) * np.sqrt(T)

    def theta(self, S, K, T, r, sigma, q=0.0, option_type="call"):
        """
        Vectorized calculation of theta for call and put options.

        Parameters:
        S (float or array-like): Current price(s) of the underlying asset
        K (float or array-like): Strike price(s)
        T (float or array-like): Time(s) to maturity (in years)
        r (float or array-like): Risk-free interest rate(s) (annual)
        sigma (float or array-like): Volatility(ies) of the underlying asset (annual)
        q (float or array-like): Continuous dividend yield(s)
        option_type (array-like of str): Option type(s) as string(s), expected to start with 'c' or 'p'

        Returns:
        np.ndarray: Theta values
        """
        option_type = self.validation_option_type(option_type)

        S     = np.asarray(S)
        K     = np.asarray(K)
        T     = np.asarray(T)
        r     = np.asarray(r)
        sigma = np.asarray(sigma)
        q     = np.asarray(q)

        d1_vals = self.d1(S, K, T, r, sigma, q)
        d2_vals = self.d2(S, K, T, r, sigma, q)
        term1   = -S * np.exp(-q*T) * norm.pdf(d1_vals) * sigma / (2 * np.sqrt(T))

        return np.where(option_type == "call",
                        term1 + q * S * np.exp(-q * T) * norm.cdf(d1_vals) - r * K * np.exp(-r * T) * norm.cdf(d2_vals),
                        term1 - q * S * np.exp(-q * T) * norm.cdf(-d1_vals) + r * K * np.exp(-r * T) * norm.cdf(-d2_vals))

    def rho(self, S, K, T, r, sigma, q=0.0 ,option_type="call"):
        """
        Vectorized calculation of rho for call and put options.

        Parameters:
        S (float or array-like): Current price(s) of the underlying asset
        K (float or array-like): Strike price(s)
        T (float or array-like): Time(s) to maturity (in years)
        r (float or array-like): Risk-free interest rate(s) (annual)
        sigma (float or array-like): Volatility(ies) of the underlying asset (annual)
        q (float or array-like): Continuous dividend yield(s)
        option_type (array-like of str): Option type(s) as string(s), expected to start with 'c' or 'p'

        Returns:
        np.ndarray: Rho values
        """
        option_type = self.validation_option_type(option_type)

        S     = np.asarray(S)
        K     = np.asarray(K)
        T     = np.asarray(T)
        r     = np.asarray(r)
        sigma = np.asarray(sigma)
        q     = np.asarray(q)

        d2_vals = self.d2(S, K, T, r, sigma, q)
        return np.where(option_type == "call",
                        K * T * np.exp(-r * T) * norm.cdf(d2_vals),
                        -K * T * np.exp(-r * T) * norm.cdf(-d2_vals))


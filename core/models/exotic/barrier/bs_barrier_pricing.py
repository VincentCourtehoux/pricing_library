import numpy as np
from scipy.stats import norm

def bs_barrier_premium(S, K, T, r, sigma, q, H, option_type, barrier_type, rebate): 
    """
    Calculate the premium of a barrier option using the Black-Scholes framework.

    Parameters:
    S (float): Current price of the underlying asset
    K (float): Strike price of the option
    T (float): Time to maturity in years
    r (float): Risk-free interest rate (annual, continuously compounded)
    sigma (float): Volatility of the underlying asset (annualized)
    q (float): Continuous dividend yield of the underlying asset
    H (float): Barrier level
    option_type (str): Type of the option, expected to be 'call' or 'put'
    barrier_type (str): Barrier style, one of 'up-in', 'up-out', 'down-in', or 'down-out'
    rebate (float): Rebate amount if the barrier is breached

    Returns:
    float: Theoretical premium of the barrier option
    """
    mu_param = (r - q - 0.5 * sigma**2) / (sigma**2)
    lambda_param = np.sqrt(mu_param**2 + 2 * r / (sigma**2))
    
    if option_type == "call":
        if barrier_type == "down-in":
            eta = 1
            phi = 1
            if K >= H:
                y1 = np.log(H**2 / (S * K)) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)
                y2 = np.log(H / S) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)

                C = phi * S * np.exp((q - r) * T) * (H / S)**(2 * (mu_param + 1)) * norm.cdf(eta * y1) - phi * K * np.exp(-r * T) * (H / S)**(2 * mu_param) * norm.cdf(eta * y1 - eta * sigma * np.sqrt(T))
                E = rebate * np.exp(-r * T) * (norm.pdf(eta * x2 - eta * sigma * np.sqrt(T)) - (H / S)**(2 * mu_param) * norm.pdf(eta * y2 - eta * np.sqrt(T)))
                return C + E 
            if K < H:
                x1 = np.log(S / K) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)
                x2 = np.log(S / H) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)
                y2 = np.log(H / S) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)
    
                A = phi * S * np.exp((q - r) * T) * norm.cdf(phi * x1) - phi * K * np.exp(-r * T) * norm.cdf(phi * x1 - phi * sigma * np.sqrt(T))
                B = phi * S * np.exp((q - r) * T) * norm.cdf(phi * x2) - phi * K * np.exp(-r * T) * norm.cdf(phi * x2 - phi * sigma * np.sqrt(T))
                D = phi * S * np.exp((q - r) * T) * (H / S)**(2 * (mu_param + 1)) * norm.cdf(eta * y2) - phi * K * np.exp(-r * T) * (H / S)**(2 * mu_param) * norm.cdf(eta * y2 - eta * sigma * np.sqrt(T))
                E = rebate * np.exp(-r * T) * (norm.pdf(eta * x2 - eta * sigma * np.sqrt(T)) - (H / S)**(2 * mu_param) * norm.pdf(eta * y2 - eta * np.sqrt(T)))
                return A - B + D + E
            
        if barrier_type == "up-in":
            eta = -1
            phi = 1
            if K >= H: 
                x1 = np.log(S / K) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)
                y2 = np.log(H / S) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)

                A = phi * S * np.exp((q - r) * T) * norm.cdf(phi * x1) - phi * K * np.exp(-r * T) * norm.cdf(phi * x1 - phi * sigma * np.sqrt(T))
                E = rebate * np.exp(-r * T) * (norm.pdf(eta * x2 - eta * sigma * np.sqrt(T)) - (H / S)**(2 * mu_param) * norm.pdf(eta * y2 - eta * np.sqrt(T)))
                return A + E
            if K < H:
                x2 = np.log(S / H) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)
                y1 = np.log(H**2 / (S * K)) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)
                y2 = np.log(H / S) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)

                B = phi * S * np.exp((q - r) * T) * norm.cdf(phi * x2) - phi * K * np.exp(-r * T) * norm.cdf(phi * x2 - phi * sigma * np.sqrt(T))
                C = phi * S * np.exp((q - r) * T) * (H / S)**(2 * (mu_param + 1)) * norm.cdf(eta * y1) - phi * K * np.exp(-r * T) * (H / S)**(2 * mu_param) * norm.cdf(eta * y1 - eta * sigma * np.sqrt(T))
                D = phi * S * np.exp((q - r) * T) * (H / S)**(2 * (mu_param + 1)) * norm.cdf(eta * y2) - phi * K * np.exp(-r * T) * (H / S)**(2 * mu_param) * norm.cdf(eta * y2 - eta * sigma * np.sqrt(T))
                E = rebate * np.exp(-r * T) * (norm.pdf(eta * x2 - eta * sigma * np.sqrt(T)) - (H / S)**(2 * mu_param) * norm.pdf(eta * y2 - eta * np.sqrt(T)))
                return B - C + D + E
            
        if barrier_type == "down-out":
            if S > H:
                eta = 1
                phi = 1
                if K >= H:
                    x1 = np.log(S / K) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)
                    y1 = np.log(H**2 / (S * K)) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)
                    z = np.log(H / K) / (sigma * np.sqrt(T)) + lambda_param * sigma * np.sqrt(T)

                    A = phi * S * np.exp((q - r) * T) * norm.cdf(phi * x1) - phi * K * np.exp(-r * T) * norm.cdf(phi * x1 - phi * sigma * np.sqrt(T))
                    C = phi * S * np.exp((q - r) * T) * (H / S)**(2 * (mu_param + 1)) * norm.cdf(eta * y1) - phi * K * np.exp(-r * T) * (H / S)**(2 * mu_param) * norm.cdf(eta * y1 - eta * sigma * np.sqrt(T))
                    F = rebate * ((H / S)**(mu_param + lambda_param) * norm.pdf(eta * z) + (H / S)**(mu_param - lambda_param) * norm.pdf(eta * z - 2 * eta * lambda_param * sigma * np.sqrt(T)))
                    return A - C + F
                if K < H:
                    x2 = np.log(S / H) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)
                    y2 = np.log(H / S) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)
                    z = np.log(H / K) / (sigma * np.sqrt(T)) + lambda_param * sigma * np.sqrt(T)

                    B = phi * S * np.exp((q - r) * T) * norm.cdf(phi * x2) - phi * K * np.exp(-r * T) * norm.cdf(phi * x2 - phi * sigma * np.sqrt(T))
                    D = phi * S * np.exp((q - r) * T) * (H / S)**(2 * (mu_param + 1)) * norm.cdf(eta * y2) - phi * K * np.exp(-r * T) * (H / S)**(2 * mu_param) * norm.cdf(eta * y2 - eta * sigma * np.sqrt(T))
                    F = rebate * ((H / S)**(mu_param + lambda_param) * norm.pdf(eta * z) + (H / S)**(mu_param - lambda_param) * norm.pdf(eta * z - 2 * eta * lambda_param * sigma * np.sqrt(T)))
                    return B - D + F
            return 0
            
        if barrier_type == "up-out":
            if S < H:
                eta = -1
                phi = 1
                if K >= H:
                    z = np.log(H / K) / (sigma * np.sqrt(T)) + lambda_param * sigma * np.sqrt(T)

                    F = rebate * ((H / S)**(mu_param + lambda_param) * norm.pdf(eta * z) + (H / S)**(mu_param - lambda_param) * norm.pdf(eta * z - 2 * eta * lambda_param * sigma * np.sqrt(T)))
                    return F
                if K < H:
                    x1 = np.log(S / K) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)
                    x2 = np.log(S / H) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)
                    y1 = np.log(H**2 / (S * K)) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)
                    y2 = np.log(H / S) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)
                    z = np.log(H / K) / (sigma * np.sqrt(T)) + lambda_param * sigma * np.sqrt(T)

                    A = phi * S * np.exp((q - r) * T) * norm.cdf(phi * x1) - phi * K * np.exp(-r * T) * norm.cdf(phi * x1 - phi * sigma * np.sqrt(T))
                    B = phi * S * np.exp((q - r) * T) * norm.cdf(phi * x2) - phi * K * np.exp(-r * T) * norm.cdf(phi * x2 - phi * sigma * np.sqrt(T))
                    C = phi * S * np.exp((q - r) * T) * (H / S)**(2 * (mu_param + 1)) * norm.cdf(eta * y1) - phi * K * np.exp(-r * T) * (H / S)**(2 * mu_param) * norm.cdf(eta * y1 - eta * sigma * np.sqrt(T))
                    D = phi * S * np.exp((q - r) * T) * (H / S)**(2 * (mu_param + 1)) * norm.cdf(eta * y2) - phi * K * np.exp(-r * T) * (H / S)**(2 * mu_param) * norm.cdf(eta * y2 - eta * sigma * np.sqrt(T))
                    F = rebate * ((H / S)**(mu_param + lambda_param) * norm.pdf(eta * z) + (H / S)**(mu_param - lambda_param) * norm.pdf(eta * z - 2 * eta * lambda_param * sigma * np.sqrt(T)))
                    return A - B + C - D + F
            else:
                return 0
            
    elif option_type == "put":
        if barrier_type == "down-in":
            eta = 1
            phi = -1
            if K >= H:
                x2 = np.log(S / H) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)
                y1 = np.log(H**2 / (S * K)) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)
                y2 = np.log(H / S) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)

                B = phi * S * np.exp((q - r) * T) * norm.cdf(phi * x2) - phi * K * np.exp(-r * T) * norm.cdf(phi * x2 - phi * sigma * np.sqrt(T))
                C = phi * S * np.exp((q - r) * T) * (H / S)**(2 * (mu_param + 1)) * norm.cdf(eta * y1) - phi * K * np.exp(-r * T) * (H / S)**(2 * mu_param) * norm.cdf(eta * y1 - eta * sigma * np.sqrt(T))
                D = phi * S * np.exp((q - r) * T) * (H / S)**(2 * (mu_param + 1)) * norm.cdf(eta * y2) - phi * K * np.exp(-r * T) * (H / S)**(2 * mu_param) * norm.cdf(eta * y2 - eta * sigma * np.sqrt(T))
                E = rebate * np.exp(-r * T) * (norm.pdf(eta * x2 - eta * sigma * np.sqrt(T)) - (H / S)**(2 * mu_param) * norm.pdf(eta * y2 - eta * np.sqrt(T)))
                return B - C + D + E
            if K < H:
                x1 = np.log(S / K) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)
                y2 = np.log(H / S) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)

                A = phi * S * np.exp((q - r) * T) * norm.cdf(phi * x1) - phi * K * np.exp(-r * T) * norm.cdf(phi * x1 - phi * sigma * np.sqrt(T))
                E = rebate * np.exp(-r * T) * (norm.pdf(eta * x2 - eta * sigma * np.sqrt(T)) - (H / S)**(2 * mu_param) * norm.pdf(eta * y2 - eta * np.sqrt(T)))
                return A + E
            
        if barrier_type == "up-in":
            eta = -1
            phi = -1
            if K >= H: 
                x1 = np.log(S / K) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)
                x2 = np.log(S / H) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)
                y2 = np.log(H / S) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)

                A = phi * S * np.exp((q - r) * T) * norm.cdf(phi * x1) - phi * K * np.exp(-r * T) * norm.cdf(phi * x1 - phi * sigma * np.sqrt(T))
                B = phi * S * np.exp((q - r) * T) * norm.cdf(phi * x2) - phi * K * np.exp(-r * T) * norm.cdf(phi * x2 - phi * sigma * np.sqrt(T))
                D = phi * S * np.exp((q - r) * T) * ((H / S)**(2 * (mu_param + 1))) * norm.cdf(eta * y2) - phi * K * np.exp(-r * T) * ((H / S)**(2 * mu_param)) * norm.cdf(eta * y2 - eta * sigma * np.sqrt(T))
                E = rebate * np.exp(-r * T) * (norm.pdf(eta * x2 - eta * sigma * np.sqrt(T)) - (H / S)**(2 * mu_param) * norm.pdf(eta * y2 - eta * np.sqrt(T)))
                return A - B + D + E
            if K < H:
                y1 = np.log(H**2 / (S * K)) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)
                y2 = np.log(H / S) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)

                C = phi * S * np.exp((q - r) * T) * (H / S)**(2 * (mu_param + 1)) * norm.cdf(eta * y1) - phi * K * np.exp(-r * T) * (H / S)**(2 * mu_param) * norm.cdf(eta * y1 - eta * sigma * np.sqrt(T))
                E = rebate * np.exp(-r * T) * (norm.pdf(eta * x2 - eta * sigma * np.sqrt(T)) - (H / S)**(2 * mu_param) * norm.pdf(eta * y2 - eta * np.sqrt(T)))
                return C + E
            
        if barrier_type == "down-out":
            if S > H:
                eta = 1
                phi = -1
                if K >= H:
                    x1 = np.log(S / K) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)
                    x2 = np.log(S / H) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)
                    y1 = np.log(H**2 / (S * K)) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)
                    y2 = np.log(H / S) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)
                    z = np.log(H / K) / (sigma * np.sqrt(T)) + lambda_param * sigma * np.sqrt(T)

                    A = phi * S * np.exp((q - r) * T) * norm.cdf(phi * x1) - phi * K * np.exp(-r * T) * norm.cdf(phi * x1 - phi * sigma * np.sqrt(T))
                    B = phi * S * np.exp((q - r) * T) * norm.cdf(phi * x2) - phi * K * np.exp(-r * T) * norm.cdf(phi * x2 - phi * sigma * np.sqrt(T))
                    C = phi * S * np.exp((q - r) * T) * (H / S)**(2 * (mu_param + 1)) * norm.cdf(eta * y1) - phi * K * np.exp(-r * T) * (H / S)**(2 * mu_param) * norm.cdf(eta * y1 - eta * sigma * np.sqrt(T))
                    D = phi * S * np.exp((q - r) * T) * (H / S)**(2 * (mu_param + 1)) * norm.cdf(eta * y2) - phi * K * np.exp(-r * T) * (H / S)**(2 * mu_param) * norm.cdf(eta * y2 - eta * sigma * np.sqrt(T))
                    F = rebate * ((H / S)**(mu_param + lambda_param) * norm.pdf(eta * z) + (H / S)**(mu_param - lambda_param) * norm.pdf(eta * z - 2 * eta * lambda_param * sigma * np.sqrt(T)))
                    return A - B + C - D + F
                if K < H:
                    z = np.log(H / K) / (sigma * np.sqrt(T)) + lambda_param * sigma * np.sqrt(T)

                    F = rebate * ((H / S)**(mu_param + lambda_param) * norm.pdf(eta * z) + (H / S)**(mu_param - lambda_param) * norm.pdf(eta * z - 2 * eta * lambda_param * sigma * np.sqrt(T)))
                    return F
            else:
                return 0
            
        if barrier_type == "up-out":
            if S < H:
                eta = -1
                phi = -1
                if K >= H:
                    x2 = np.log(S / H) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)
                    y2 = np.log(H / S) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)
                    z = np.log(H / K) / (sigma * np.sqrt(T)) + lambda_param * sigma * np.sqrt(T)

                    B = phi * S * np.exp((q - r) * T) * norm.cdf(phi * x2) - phi * K * np.exp(-r * T) * norm.cdf(phi * x2 - phi * sigma * np.sqrt(T))
                    D = phi * S * np.exp((q - r) * T) * (H / S)**(2 * (mu_param + 1)) * norm.cdf(eta * y2) - phi * K * np.exp(-r * T) * (H / S)**(2 * mu_param) * norm.cdf(eta * y2 - eta * sigma * np.sqrt(T))
                    F = rebate * ((H / S)**(mu_param + lambda_param) * norm.pdf(eta * z) + (H / S)**(mu_param - lambda_param) * norm.pdf(eta * z - 2 * eta * lambda_param * sigma * np.sqrt(T)))
                    return B - D + F
                if K < H:
                    x1 = np.log(S / K) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)
                    y1 = np.log(H**2 / (S * K)) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)
                    z = np.log(H / K) / (sigma * np.sqrt(T)) + lambda_param * sigma * np.sqrt(T)

                    A = phi * S * np.exp((q - r) * T) * norm.cdf(phi * x1) - phi * K * np.exp(-r * T) * norm.cdf(phi * x1 - phi * sigma * np.sqrt(T))
                    C = phi * S * np.exp((q - r) * T) * (H / S)**(2 * (mu_param + 1)) * norm.cdf(eta * y1) - phi * K * np.exp(-r * T) * (H / S)**(2 * mu_param) * norm.cdf(eta * y1 - eta * sigma * np.sqrt(T))
                    F = rebate * ((H / S)**(mu_param + lambda_param) * norm.pdf(eta * z) + (H / S)**(mu_param - lambda_param) * norm.pdf(eta * z - 2 * eta * lambda_param * sigma * np.sqrt(T)))
                    return A - C
            else:
                return 0
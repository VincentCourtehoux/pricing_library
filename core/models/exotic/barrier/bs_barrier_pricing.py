import numpy as np
from scipy.stats import norm

def bs_barrier_premium(S, K, T, r, sigma, q, H, option_type, barrier_type):  

    mu_param = (r - q - 0.5 * sigma**2) / (sigma**2)
    lambda_param = np.sqrt(mu_param**2 + 2 * r / (sigma**2))

    if option_type == "call":
        if barrier_type == "down-in":
            eta = 1
            phi = 1
            if K > H:
                y1 = np.log(H**2 / (S * K)) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)

                C = phi * S * np.exp((q - r) * T) * (H / S)**(2 * (mu_param + 1)) * norm.cdf(eta * y1) - phi * K * np.exp(-r * T) * (H / S)**(2 * mu_param) * norm.cdf(eta * y1 - eta * sigma * np.sqrt(T))
                return C
            if K < H:
                x1 = np.log(S / K) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)
                x2 = np.log(S / H) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)
                y2 = np.log(H / S) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)
    
                A = phi * S * np.exp((q - r) * T) * norm.cdf(phi * x1) - phi * K * np.exp(-r * T) * norm.cdf(phi * x1 - phi * sigma * np.sqrt(T))
                B = phi * S * np.exp((q - r) * T) * norm.cdf(phi * x2) - phi * K * np.exp(-r * T) * norm.cdf(phi * x2 - phi * sigma * np.sqrt(T))
                D = phi * S * np.exp((q - r) * T) * (H / S)**(2 * (mu_param + 1)) * norm.cdf(eta * y2) - phi * K * np.exp(-r * T) * (H / S)**(2 * mu_param) * norm.cdf(eta * y2 - eta * sigma * np.sqrt(T))

                return A - B + D
            
        if barrier_type == "up-in":
            eta = -1
            phi = 1
            if K > H: 
                x1 = np.log(S / K) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)

                A = phi * S * np.exp((q - r) * T) * norm.cdf(phi * x1) - phi * K * np.exp(-r * T) * norm.cdf(phi * x1 - phi * sigma * np.sqrt(T))
                return A
            if K < H:
                x2 = np.log(S / H) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)
                y1 = np.log(H**2 / (S * K)) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)
                y2 = np.log(H / S) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)

                B = phi * S * np.exp((q - r) * T) * norm.cdf(phi * x2) - phi * K * np.exp(-r * T) * norm.cdf(phi * x2 - phi * sigma * np.sqrt(T))
                C = phi * S * np.exp((q - r) * T) * (H / S)**(2 * (mu_param + 1)) * norm.cdf(eta * y1) - phi * K * np.exp(-r * T) * (H / S)**(2 * mu_param) * norm.cdf(eta * y1 - eta * sigma * np.sqrt(T))
                D = phi * S * np.exp((q - r) * T) * (H / S)**(2 * (mu_param + 1)) * norm.cdf(eta * y2) - phi * K * np.exp(-r * T) * (H / S)**(2 * mu_param) * norm.cdf(eta * y2 - eta * sigma * np.sqrt(T))
                return B - C + D
            
        if barrier_type == "down-out":
            if S > H:
                eta = 1
                phi = 1
                if K > H:
                    x1 = np.log(S / K) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)
                    y1 = np.log(H**2 / (S * K)) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)

                    A = phi * S * np.exp((q - r) * T) * norm.cdf(phi * x1) - phi * K * np.exp(-r * T) * norm.cdf(phi * x1 - phi * sigma * np.sqrt(T))
                    C = phi * S * np.exp((q - r) * T) * (H / S)**(2 * (mu_param + 1)) * norm.cdf(eta * y1) - phi * K * np.exp(-r * T) * (H / S)**(2 * mu_param) * norm.cdf(eta * y1 - eta * sigma * np.sqrt(T))
                    return A - C
                if K < H:
                    x2 = np.log(S / H) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)
                    y2 = np.log(H / S) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)

                    B = phi * S * np.exp((q - r) * T) * norm.cdf(phi * x2) - phi * K * np.exp(-r * T) * norm.cdf(phi * x2 - phi * sigma * np.sqrt(T))
                    D = phi * S * np.exp((q - r) * T) * (H / S)**(2 * (mu_param + 1)) * norm.cdf(eta * y2) - phi * K * np.exp(-r * T) * (H / S)**(2 * mu_param) * norm.cdf(eta * y2 - eta * sigma * np.sqrt(T))
                    return B - D
            return 0
            
        if barrier_type == "up-out":
            if S < H:
                eta = -1
                phi = 1
                if K > H:
                    return 0
                if K < H:
                    x1 = np.log(S / K) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)
                    x2 = np.log(S / H) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)
                    y1 = np.log(H**2 / (S * K)) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)
                    y2 = np.log(H / S) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)

                    A = phi * S * np.exp((q - r) * T) * norm.cdf(phi * x1) - phi * K * np.exp(-r * T) * norm.cdf(phi * x1 - phi * sigma * np.sqrt(T))
                    B = phi * S * np.exp((q - r) * T) * norm.cdf(phi * x2) - phi * K * np.exp(-r * T) * norm.cdf(phi * x2 - phi * sigma * np.sqrt(T))
                    C = phi * S * np.exp((q - r) * T) * (H / S)**(2 * (mu_param + 1)) * norm.cdf(eta * y1) - phi * K * np.exp(-r * T) * (H / S)**(2 * mu_param) * norm.cdf(eta * y1 - eta * sigma * np.sqrt(T))
                    D = phi * S * np.exp((q - r) * T) * (H / S)**(2 * (mu_param + 1)) * norm.cdf(eta * y2) - phi * K * np.exp(-r * T) * (H / S)**(2 * mu_param) * norm.cdf(eta * y2 - eta * sigma * np.sqrt(T))
                    return A - B + C - D
            else:
                return 0
            
    elif option_type == "put":
        if barrier_type == "down-in":
            eta = 1
            phi = -1
            if K > H:
                x2 = np.log(S / H) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)
                y1 = np.log(H**2 / (S * K)) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)
                y2 = np.log(H / S) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)

                B = phi * S * np.exp((q - r) * T) * norm.cdf(phi * x2) - phi * K * np.exp(-r * T) * norm.cdf(phi * x2 - phi * sigma * np.sqrt(T))
                C = phi * S * np.exp((q - r) * T) * (H / S)**(2 * (mu_param + 1)) * norm.cdf(eta * y1) - phi * K * np.exp(-r * T) * (H / S)**(2 * mu_param) * norm.cdf(eta * y1 - eta * sigma * np.sqrt(T))
                D = phi * S * np.exp((q - r) * T) * (H / S)**(2 * (mu_param + 1)) * norm.cdf(eta * y2) - phi * K * np.exp(-r * T) * (H / S)**(2 * mu_param) * norm.cdf(eta * y2 - eta * sigma * np.sqrt(T))
                return B - C + D
            if K < H:
                x1 = np.log(S / K) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)

                A = phi * S * np.exp((q - r) * T) * norm.cdf(phi * x1) - phi * K * np.exp(-r * T) * norm.cdf(phi * x1 - phi * sigma * np.sqrt(T))
                return A
            
        if barrier_type == "up-in":
            eta = -1
            phi = -1
            if K > H: 
                x1 = np.log(S / K) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)
                x2 = np.log(S / H) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)
                y2 = np.log(H / S) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)

                A = phi * S * np.exp((q - r) * T) * norm.cdf(phi * x1) - phi * K * np.exp(-r * T) * norm.cdf(phi * x1 - phi * sigma * np.sqrt(T))
                B = phi * S * np.exp((q - r) * T) * norm.cdf(phi * x2) - phi * K * np.exp(-r * T) * norm.cdf(phi * x2 - phi * sigma * np.sqrt(T))
                D = phi * S * np.exp((q - r) * T) * ((H / S)**(2 * (mu_param + 1))) * norm.cdf(eta * y2) - phi * K * np.exp(-r * T) * ((H / S)**(2 * mu_param)) * norm.cdf(eta * y2 - eta * sigma * np.sqrt(T))
                return A - B + D
            if K < H:
                y1 = np.log(H**2 / (S * K)) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)

                C = phi * S * np.exp((q - r) * T) * (H / S)**(2 * (mu_param + 1)) * norm.cdf(eta * y1) - phi * K * np.exp(-r * T) * (H / S)**(2 * mu_param) * norm.cdf(eta * y1 - eta * sigma * np.sqrt(T))
                print("C: ", C)
                print("y1: ", y1)
                return C
            
        if barrier_type == "down-out":
            if S > H:
                eta = 1
                phi = -1
                if K > H:
                    x1 = np.log(S / K) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)
                    x2 = np.log(S / H) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)
                    y1 = np.log(H**2 / (S * K)) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)
                    y2 = np.log(H / S) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)

                    A = phi * S * np.exp((q - r) * T) * norm.cdf(phi * x1) - phi * K * np.exp(-r * T) * norm.cdf(phi * x1 - phi * sigma * np.sqrt(T))
                    B = phi * S * np.exp((q - r) * T) * norm.cdf(phi * x2) - phi * K * np.exp(-r * T) * norm.cdf(phi * x2 - phi * sigma * np.sqrt(T))
                    C = phi * S * np.exp((q - r) * T) * (H / S)**(2 * (mu_param + 1)) * norm.cdf(eta * y1) - phi * K * np.exp(-r * T) * (H / S)**(2 * mu_param) * norm.cdf(eta * y1 - eta * sigma * np.sqrt(T))
                    D = phi * S * np.exp((q - r) * T) * (H / S)**(2 * (mu_param + 1)) * norm.cdf(eta * y2) - phi * K * np.exp(-r * T) * (H / S)**(2 * mu_param) * norm.cdf(eta * y2 - eta * sigma * np.sqrt(T))
                    return A - B + C - D
                if K < H:
                    return 0
            else:
                return 0
            
        if barrier_type == "up-out":
            if S < H:
                eta = -1
                phi = -1
                if K > H:
                    x2 = np.log(S / H) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)
                    y2 = np.log(H / S) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)

                    B = phi * S * np.exp((q - r) * T) * norm.cdf(phi * x2) - phi * K * np.exp(-r * T) * norm.cdf(phi * x2 - phi * sigma * np.sqrt(T))
                    D = phi * S * np.exp((q - r) * T) * (H / S)**(2 * (mu_param + 1)) * norm.cdf(eta * y2) - phi * K * np.exp(-r * T) * (H / S)**(2 * mu_param) * norm.cdf(eta * y2 - eta * sigma * np.sqrt(T))
                    return B - D 
                if K < H:
                    x1 = np.log(S / K) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)
                    y1 = np.log(H**2 / (S * K)) / (sigma * np.sqrt(T)) + (1 + mu_param) * sigma * np.sqrt(T)

                    A = phi * S * np.exp((q - r) * T) * norm.cdf(phi * x1) - phi * K * np.exp(-r * T) * norm.cdf(phi * x1 - phi * sigma * np.sqrt(T))
                    C = phi * S * np.exp((q - r) * T) * (H / S)**(2 * (mu_param + 1)) * norm.cdf(eta * y1) - phi * K * np.exp(-r * T) * (H / S)**(2 * mu_param) * norm.cdf(eta * y1 - eta * sigma * np.sqrt(T))
                    return A - C
            else:
                return 0
        
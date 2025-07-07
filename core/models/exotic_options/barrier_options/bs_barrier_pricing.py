import numpy as np
from scipy.stats import norm
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from vanilla_options.european_options.black_scholes.pricing_scalar import BlackScholesScalar
bs = BlackScholesScalar()


def bs_barrier_premium(S, K, T, r, sigma, q, H, option_type, barrier_type):
    lambda_param = (r - q + 0.5 * sigma**2) / sigma**2
    y = np.log(H**2 / (S * K)) / (sigma * np.sqrt(T)) + lambda_param * sigma * np.sqrt(T)
    x1 = np.log(S / H) / (sigma * np.sqrt(T)) + lambda_param * sigma * np.sqrt(T)
    y1 = np.log(H / S) / (sigma * np.sqrt(T)) + lambda_param * sigma * np.sqrt(T)
    if option_type == "call":
        if barrier_type in ["down-in", "down-out"]:
            if S > H:
                if H <= K:
                    price_di_call = S * np.exp(-q * T) * (H / S)**(2 * lambda_param) * norm.cdf(y) - K * np.exp(
                        -r * T) * (H / S)**(2 * lambda_param - 2) * norm.cdf(y - sigma * np.sqrt(T))
                    price_do_call = bs.bs_eu_scalar_premium(S, K, T, r, sigma, q, option_type='call') - price_di_call

                if H >= K:
                    price_do_call = S * norm.cdf(x1) * np.exp(-q * T) - K * np.exp(-r * T) * norm.cdf(
                        x1 - sigma * np.sqrt(T)) - S * np.exp(-q * T) * (H / S)**(2 * lambda_param) * norm.cdf(
                        y1) + K * np.exp(-r * T) * (H / S)**(2 * lambda_param - 2) * norm.cdf(y1 - sigma * np.sqrt(T))
                    price_di_call = bs.bs_eu_scalar_premium(S, K, T, r, sigma, q, option_type) - price_do_call
            else:
                price_do_call = 0
                price_di_call = bs.bs_eu_scalar_premium(S, K, T, r, sigma, q, option_type)
            return price_do_call if barrier_type == "down-out" else price_di_call
        if barrier_type in ["up-out", "up-in"]:
            if S < H:
                if H <= K:
                    price_uo_call = 0
                    price_ui_call = bs.bs_eu_scalar_premium(S, K, T, r, sigma, q, option_type)
                if H > K:
                    price_ui_call = S * norm.cdf(x1) * np.exp(-q * T) - K * np.exp(-r * T) * norm.cdf(
                        x1 - sigma * np.sqrt(T)) - S * np.exp(-q * T) * (H / S)**(2 * lambda_param) * (
                        norm.cdf(-y) - norm.cdf(-y1)) + K * np.exp(-r * T) * (H / S)**(2 * lambda_param - 2) * (
                        norm.cdf(-y + sigma * np.sqrt(T)) - norm.cdf(-y1 + sigma * np.sqrt(T)))
                    price_uo_call = bs.bs_eu_scalar_premium(S, K, T, r, sigma, q, option_type='call') - price_ui_call
            else:
                price_uo_call = 0
                price_ui_call = bs.bs_eu_scalar_premium(S, K, T, r, sigma, q, option_type)
            return price_uo_call if barrier_type == "up-out" else price_ui_call
    if option_type == "put":
        if barrier_type in ["up-out", "up-in"]:
            if S < H:
                if H >= K:
                    price_ui_put = -S * np.exp(-q * T) * (H / S)**(2 * lambda_param) * norm.cdf(-y) + K * np.exp(
                        -r * T) * (H / S)**(2 * lambda_param - 2) * norm.cdf(-y + sigma * np.sqrt(T))
                    price_uo_put = bs.bs_eu_scalar_premium(S, K, T, r, sigma, q, option_type='put') - price_ui_put
                if H < K:
                    price_uo_put = -S * norm.cdf(-x1) * np.exp(-q * T) + K * np.exp(-r * T) * norm.cdf(
                        -x1 + sigma * np.sqrt(T)) + S * np.exp(-q * T) * (H / S)**(2 * lambda_param) * norm.cdf(
                        -y1) - K * np.exp(-r * T) * (H / S) * (2 * lambda_param - 2) * norm.cdf(-y1 + sigma * np.sqrt(T))
                    price_ui_put = bs.bs_eu_scalar_premium(S, K, T, r, sigma, q, option_type='put') - price_uo_put
            else:
                price_uo_put = 0
                price_ui_put = bs.bs_eu_scalar_premium(S, K, T, r, sigma, q, option_type)
            return price_uo_put if barrier_type == "up-out" else price_ui_put   
        if barrier_type in ["down-out", "down-in"]:
            if S > H:  
                if H > K:
                    price_do_put = 0
                    price_di_put = bs.bs_eu_scalar_premium(S, K, T, r, sigma, q, option_type='put')
                if H <= K:
                    price_di_put = -S * norm.cdf(-x1) * np.exp(-q * T) + K * np.exp(-r * T) * norm.cdf(-x1 + sigma * np.sqrt(T)) + S * np.exp(-q * T) * (H / S)**(2 * lambda_param) * (norm.cdf(y) - norm.cdf(y1)) - K * np.exp(-r * T) * (H / S)**(2 * lambda_param - 2) * (norm.cdf(y - sigma * np.sqrt(T)) - norm.cdf(y1 - sigma * np.sqrt(T)))
                    price_do_put = bs.bs_eu_scalar_premium(S, K, T, r, sigma, q, option_type='put') - price_di_put
            else: 
                price_do_put = 0
                price_di_put = bs.bs_eu_scalar_premium(S, K, T, r, sigma, q, option_type)
            return price_do_put if barrier_type == "down-out" else price_di_put
import numpy as np
from scipy.stats import norm
from .black_scholes import BlackScholesModel
from .base_model import PricingModel


class BlackScholesBarrierModel(PricingModel):

    def calculate(self, S, K, T, r, sigma, option_type='call', q=0.0, **kwargs):
        barrier_type = kwargs.get('barrier_type', 'down-and-out')
        barrier_level = kwargs.get('barrier_level', 0)
        q = kwargs.get('q', 0)

        if barrier_type is None:
            raise ValueError("barrier_type is required for barrier options")
        if barrier_level is None:
            raise ValueError("barrier_level is required for barrier options")

        bs_model = BlackScholesModel()
        vanilla_price = bs_model.calculate(S, K, T, r, sigma, option_type)['price']

        lambda_param = (r - q + 0.5 * sigma**2) / sigma**2
        y = np.log(barrier_level**2 / (S * K)) / (sigma * np.sqrt(T)) + lambda_param * sigma * np.sqrt(T)
        x1 = np.log(S / barrier_level) / (sigma * np.sqrt(T)) + lambda_param * sigma * np.sqrt(T)
        y1 = np.log(barrier_level / S) / (sigma * np.sqrt(T)) + lambda_param * sigma * np.sqrt(T)

        if option_type == "call":
            if barrier_type in ["down-and-in", "down-and-out"]:
                if S > barrier_level:
                    if barrier_level <= K:
                        price_di = S * np.exp(-q * T) * (barrier_level / S)**(2 * lambda_param) * norm.cdf(y) - K * np.exp(
                            -r * T) * (barrier_level / S)**(2 * lambda_param - 2) * norm.cdf(y - sigma * np.sqrt(T))
                        price_do = vanilla_price - price_di

                    else:
                        price_do = S * norm.cdf(x1) * np.exp(-q * T) - K * np.exp(-r * T) * norm.cdf(
                            x1 - sigma * np.sqrt(T)) - S * np.exp(-q * T) * (barrier_level / S)**(2 * lambda_param) * norm.cdf(
                            y1) + K * np.exp(-r * T) * (barrier_level / S)**(2 * lambda_param - 2) * norm.cdf(y1 - sigma * np.sqrt(T))
                        price_di = vanilla_price - price_do
                else:
                    price_do = 0
                    price_di = vanilla_price
                final_price = price_do if barrier_type == "down-and-out" else price_di
            if barrier_type in ["up-and-out", "up-and-in"]:
                if S < barrier_level:
                    if barrier_level <= K:
                        price_uo = 0
                        price_ui = vanilla_price
                    else:
                        price_ui = S * norm.cdf(x1) * np.exp(-q * T) - K * np.exp(-r * T) * norm.cdf(
                            x1 - sigma * np.sqrt(T)) - S * np.exp(-q * T) * (barrier_level / S)**(2 * lambda_param) * (
                            norm.cdf(-y) - norm.cdf(-y1)) + K * np.exp(-r * T) * (barrier_level / S)**(2 * lambda_param - 2) * (
                            norm.cdf(-y + sigma * np.sqrt(T)) - norm.cdf(-y1 + sigma * np.sqrt(T)))
                        price_uo = vanilla_price - price_ui
                else:
                    price_uo = 0
                    price_ui = vanilla_price
                final_price = price_uo if barrier_type == "up-and-out" else price_ui
        if option_type == "put":
            if barrier_type in ["up-and-out", "up-and-in"]:
                if S < barrier_level:
                    if barrier_level >= K:
                        price_ui = -S * np.exp(-q * T) * (barrier_level / S)**(2 * lambda_param) * norm.cdf(-y) + K * np.exp(
                            -r * T) * (barrier_level / S)**(2 * lambda_param - 2) * norm.cdf(-y + sigma * np.sqrt(T))
                        price_uo = vanilla_price - price_ui
                    else:
                        price_uo = -S * norm.cdf(-x1) * np.exp(-q * T) + K * np.exp(-r * T) * norm.cdf(
                            -x1 + sigma * np.sqrt(T)) + S * np.exp(-q * T) * (barrier_level / S)**(2 * lambda_param) * norm.cdf(
                            -y1) - K * np.exp(-r * T) * (barrier_level / S) ** (2 * lambda_param - 2) * norm.cdf(-y1 + sigma * np.sqrt(T))
                        price_ui = vanilla_price - price_uo
                else:
                    price_uo = 0
                    price_ui = vanilla_price
                final_price = price_uo if barrier_type == "up-and-out" else price_ui   
            if barrier_type in ["down-and-out", "down-and-in"]:
                if S > barrier_level:  
                    if barrier_level > K:
                        price_do = 0
                        price_di = vanilla_price
                    else:
                        price_di = -S * norm.cdf(-x1) * np.exp(-q * T) + K * np.exp(-r * T) * norm.cdf(
                            -x1 + sigma * np.sqrt(T)) + S * np.exp(-q * T) * (barrier_level / S)**(2 * lambda_param) * (
                            norm.cdf(y) - norm.cdf(y1)) - K * np.exp(-r * T) * (barrier_level / S)**(2 * lambda_param - 2) * (
                            norm.cdf(y - sigma * np.sqrt(T)) - norm.cdf(y1 - sigma * np.sqrt(T)))
                        price_do = vanilla_price - price_di
                else: 
                    price_do = 0
                    price_di = vanilla_price
                final_price = price_do if barrier_type == "down-and-out" else price_di

        return {
            'price': final_price,
            'method': 'black_scholes',
            'option_style': 'barrier'
        }
        
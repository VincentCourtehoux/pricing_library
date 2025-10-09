import numpy as np
from scipy.stats import norm
from .base_model import PricingModel

class BlackScholesModel(PricingModel):
    def calculate(self, S, K, T, r, sigma, option_type='call', q=0.0, **kwargs):
        S = np.atleast_1d(S).astype(float)
        K = np.atleast_1d(K).astype(float)
        T = np.atleast_1d(T).astype(float)
        r = np.atleast_1d(r).astype(float)
        sigma = np.atleast_1d(sigma).astype(float)
        q = np.atleast_1d(q).astype(float)
        option_type = np.atleast_1d(option_type)

        price = np.zeros_like(S)
        expired = T <= 0
        alive = ~expired

        if np.any(expired):
            call_payoff = np.maximum(S[expired] - K[expired], 0)
            put_payoff  = np.maximum(K[expired] - S[expired], 0)
            price[expired] = np.where(option_type[expired]=='call', call_payoff, put_payoff)

        if np.any(alive):
            d1 = (np.log(S[alive]/K[alive]) + (r[alive] - q[alive] + 0.5*sigma[alive]**2)*T[alive]) / (sigma[alive]*np.sqrt(T[alive]))
            d2 = d1 - sigma[alive]*np.sqrt(T[alive])
            
            call_price = S[alive]*np.exp(-q[alive]*T[alive])*norm.cdf(d1) - K[alive]*np.exp(-r[alive]*T[alive])*norm.cdf(d2)
            put_price  = K[alive]*np.exp(-r[alive]*T[alive])*norm.cdf(-d2) - S[alive]*np.exp(-q[alive]*T[alive])*norm.cdf(-d1)
            
            price[alive] = np.where(option_type[alive]=='call', call_price, put_price)

        return {
            'price': price,
            'method': 'black_scholes',
            'option_style': 'european'
        }

    def calculate_greeks(self, S, K, T, r, sigma, option_type='call', q=0.0, **kwargs):
        S = np.atleast_1d(S).astype(float)
        K = np.atleast_1d(K).astype(float)
        T = np.atleast_1d(T).astype(float)
        r = np.atleast_1d(r).astype(float)
        sigma = np.atleast_1d(sigma).astype(float)
        q = np.atleast_1d(q).astype(float)
        option_type = np.atleast_1d(option_type)

        n = len(S)
        delta = np.zeros(n)
        gamma = np.zeros(n)
        vega = np.zeros(n)
        theta = np.zeros(n)
        rho = np.zeros(n)

        expired = T <= 0
        alive = ~expired

        if np.any(expired):
            delta[expired] = np.where(
                option_type[expired]=='call', 
                (S[expired]>K[expired]).astype(float), 
                (S[expired]<K[expired]).astype(float)
            )
        
        if np.any(alive):
            d1 = (np.log(S[alive]/K[alive]) + (r[alive]-q[alive]+0.5*sigma[alive]**2)*T[alive]) / (sigma[alive]*np.sqrt(T[alive]))
            d2 = d1 - sigma[alive]*np.sqrt(T[alive])

            price = S[alive]*np.exp(-q[alive]*T[alive])*norm.cdf(d1) - K[alive]*np.exp(-r[alive]*T[alive])*norm.cdf(d2)
            if np.any(option_type[alive]=='put'):
                put_mask = option_type[alive]=='put'
                price[put_mask] = K[alive][put_mask]*np.exp(-r[alive][put_mask]*T[alive][put_mask])*norm.cdf(-d2[put_mask]) - S[alive][put_mask]*np.exp(-q[alive][put_mask]*T[alive][put_mask])*norm.cdf(-d1[put_mask])

            delta[alive] = np.where(option_type[alive]=='call', np.exp(-q[alive]*T[alive])*norm.cdf(d1), np.exp(-q[alive]*T[alive])*(norm.cdf(d1)-1))
            gamma[alive] = np.exp(-q[alive]*T[alive])*norm.pdf(d1)/(S[alive]*sigma[alive]*np.sqrt(T[alive]))
            vega[alive] = S[alive]*np.exp(-q[alive]*T[alive])*norm.pdf(d1)*np.sqrt(T[alive])/100
            theta[alive] = (-S[alive]*np.exp(-q[alive]*T[alive])*norm.pdf(d1)*sigma[alive]/(2*np.sqrt(T[alive]))
                            - r[alive]*K[alive]*np.exp(-r[alive]*T[alive])*norm.cdf(d2)
                            + q[alive]*S[alive]*np.exp(-q[alive]*T[alive])*norm.cdf(d1))/365
            rho[alive] = K[alive]*T[alive]*np.exp(-r[alive]*T[alive])*norm.cdf(d2)/100
            rho[alive][option_type[alive]=='put'] *= -1

        return {
            'price': self.calculate(S,K,T,r,sigma,option_type,q)['price'],
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho
        }

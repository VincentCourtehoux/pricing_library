import numpy as np
import sys
import os
from numpy.polynomial.laguerre import lagval

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from models.utils.path_simulation import simulate_gbm

def payoff(S, K, option_type):
    if option_type == 'call':
        return np.maximum(S - K, 0)
    elif option_type == 'put':
        return np.maximum(K - S, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
def laguerre_matrix(x, degree):
    # x : vecteur 1D
    # Retourne une matrice (len(x), degree+1) où col i = L_i(x)
    # numpy polynômes laguerre attend les coefficients, on fait un for pour chaque degré
    mat = np.zeros((len(x), degree+1))
    for i in range(degree+1):
        c = np.zeros(i+1)
        c[-1] = 1  # coefficient 1 pour le terme L_i
        mat[:, i] = lagval(x, c)
    return mat

def longstaff_schwartz_american(S0, K, r, sigma, T, q, N, nb_paths, option_type='call', degree=2, seed=None):
    """
    Prix d'une option américaine avec dividendes via Longstaff-Schwartz
    
    option_type: 'call' ou 'put'
    """
    dt = T / N
    discount = np.exp(-r * dt)
    
    S = simulate_gbm(S0, r, sigma, T, q, N, nb_paths, seed)
    
    CF = payoff(S[-1], K, option_type)
    
    for t in range(N-1, 0, -1):
        itm = payoff(S[t], K, option_type) > 0
        S_itm = S[t, itm]
        CF_itm = CF[itm] * discount
        
        if len(S_itm) == 0:
            CF = CF * discount
            continue
        
        X = laguerre_matrix(S_itm, degree)
        coeffs = np.linalg.lstsq(X, CF_itm, rcond=None)[0]
        continuation_value = X @ coeffs
        
        exercise_value = payoff(S_itm, K, option_type)
        exercise = exercise_value > continuation_value
        
        CF[itm] = np.where(exercise, exercise_value, CF_itm)
        CF = CF * discount
    
    price = np.mean(CF)
    return price

if __name__ == "__main__":
    S0 = 13.5
    K = 13
    r = 0.0213
    sigma = 0.427
    T = 3
    q = 0.012
    N = 200
    nb_paths = 100000

    prix_call = longstaff_schwartz_american(S0, K, r, sigma, T, q, N, nb_paths, option_type='call')
    prix_put = longstaff_schwartz_american(S0, K, r, sigma, T, q, N, nb_paths, option_type='put')

    print(f"Prix Call Américain (dividendes) : {prix_call:.4f}")
    print(f"Prix Put Américain (dividendes)  : {prix_put:.4f}")
    
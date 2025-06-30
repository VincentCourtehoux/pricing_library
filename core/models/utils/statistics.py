import numpy as np
from scipy.stats import norm

def confidence_interval(data, alpha=0.05):
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    n = len(data)
    z = norm.ppf(1 - alpha / 2)
    margin = z * std / np.sqrt(n)
    return mean, margin
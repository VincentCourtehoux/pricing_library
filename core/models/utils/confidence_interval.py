import numpy as np
from scipy.stats import norm, t

def confidence_interval(data, alpha=0.05):
    """
    Computes the (1 - alpha)% confidence interval of the mean of the data.

    Parameters:
    data (np.ndarray): Array of sample values
    alpha (float): Significance level (default 0.05 for 95% confidence)

    Returns:
    tuple: (mean, margin of error)
    """
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    n = len(data)

    z = norm.ppf(1 - alpha / 2)
    margin = z * std / np.sqrt(n)
    
    return mean, margin
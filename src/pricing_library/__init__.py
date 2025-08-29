"""
Pricing Library - A comprehensive library for financial option pricing.

This library provides various models for pricing European, Asian, and American options
using methods like Black-Scholes, Monte Carlo, Binomial Trees, and Least Squares Monte Carlo.
"""

from .core.pricing_service import PricingService
from .core.exceptions import UnsupportedOptionTypeError

from .models import (
    BlackScholesModel,
    VanillaMonteCarlo,
    BinomialModel,
    LeastSquaresMC,
    BlackScholesGapModel,
    BlackScholesBarrierModel
)

from .calculators import (
    EuropeanCalculator,
    AsianCalculator,
    AmericanCalculator,
    GapCalculator,
    BarrierCalculator
)

from .utils import (
    GeometricBrownianMotion,
    get_regressor,
    asian_payoff,
    intrinsic_value
)

__version__ = "1.0.0"
__author__ = "Vincent Courtehoux"
__email__ = "vincentcourtehoux@gmail.com"

__all__ = [
    # Core components
    'PricingService',
    'UnsupportedOptionTypeError',
    
    # Models
    'BlackScholesModel',
    'VanillaMonteCarlo', 
    'BinomialModel',
    'LeastSquaresMC',
    'BlackScholesGapModel',
    'BlackScholesBarrierModel',
    
    # Calculators
    'EuropeanCalculator',
    'AsianCalculator',
    'AmericanCalculator',
    'GapCalculator',
    'BarrierCalculator',
    
    # Utilities
    'GeometricBrownianMotion',
    'get_regressor',
    'asian_payoff',
    'intrinsic_value'
]
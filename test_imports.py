# test_pricing_library.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pricing_library.models.implied_volatility import ImpliedVolatility



print(ImpliedVolatility.calculate(10.45,100,100,1,0.05))
from pricing_library.calculators import (
    EuropeanCalculator,
    AsianCalculator,
    AmericanCalculator,
    GapCalculator,
    BarrierCalculator,
    OptionCombinationCalculator
)
from .exceptions import UnsupportedOptionTypeError

class PricingService:
    def __init__(self, calculators=None):
        self.calculators = calculators or {
            'european': EuropeanCalculator(),
            'asian': AsianCalculator(),
            'american': AmericanCalculator(),
            'gap': GapCalculator(),
            'barrier': BarrierCalculator(),
            'combination': OptionCombinationCalculator()
        }

    def price_option(self, params, method=None):
        option_style = params.get('option_style', 'european')
        calculator = self.calculators.get(option_style)

        if not calculator:
            raise UnsupportedOptionTypeError(option_style)
        
        if method is None:
            return calculator.calculate(params)
        
        return calculator.calculate(params, method)
    
    def calculate_greeks(self, params, method=None):
        option_style = params.get('option_style', 'european')
        calculator = self.calculators.get(option_style)

        if not calculator:
            raise UnsupportedOptionTypeError(option_style)
        
        if method is None:
            return calculator.calculate_greeks(params)
        
        return calculator.calculate_greeks(params, method)
class UnsupportedOptionTypeError(Exception):
    def __init__(self, option_type):
        super().__init__(f"Unsupported option type: {option_type}")
        self.option_type = option_type

class PricingError(Exception):
    pass
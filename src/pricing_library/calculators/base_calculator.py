from abc import ABC, abstractmethod

class BaseCalculator(ABC):
    
    @abstractmethod
    def calculate(self, params, method):
        pass

    @abstractmethod
    def calculate_greeks(self, params, method=None):
        pass
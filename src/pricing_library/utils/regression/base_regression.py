from abc import ABC, abstractmethod

class BaseRegression(ABC):
    
    @abstractmethod
    def fit_predict(self, X, y, X_pred):
        pass
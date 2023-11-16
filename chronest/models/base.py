from abc import ABC, abstractmethod

class Model(ABC):
    
    """
    Abstract class for time series model

    """

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass


        

    
    
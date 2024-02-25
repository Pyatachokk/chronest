from abc import ABC, abstractmethod


class BaseModel(ABC):

    @abstractmethod
    def fit():
        pass

    @abstractmethod
    def predict():
        pass

from abc import ABC, abstractmethod
from chronest.utils.constants import DATE_COLUMN
import operator
import polars as pl

class Model(ABC):
    
    """
    Abstract class for time series model

    """

    y = property(operator.attrgetter('_y'))

    @y.setter
    def y(self, d: pl.DataFrame):

        if d is None or d.is_empty(): raise Exception("description cannot be empty")

        if not d.find_idx_by_name(DATE_COLUMN): raise ValueError(f"Column {DATE_COLUMN} is not found.")

        if type(d) is not pl.DataFrame: raise TypeError("Type of y is invalid.")

        self._y = d

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass


        

    
    
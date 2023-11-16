from chronest.models.base import Model
import polars as pl
from typing import Union
pl.TEMPORAL_DTYPES



class Naive(Model):

    def __init__():
        pass 

    def fit(self, y: pl.DataFrame):
        """
        fit naive model

        Arguments:
            y {pl.DataFrame} -- Input data frame. Has to include column "date".

        """

        self._y = y
        
    def predict(h: int):
        

        pass
import pandas as pd

from chronest.univariate.models.base import BaseModel


class Naive(BaseModel):

    def __init__(
        self,
        delta: pd.DateOffset,
        seasonal_period: int = 1,
    ):

        assert type(delta) is pd.DateOffset
        self._delta = delta
        self._seasonal_period = seasonal_period

    def fit(self, y: pd.Series):

        self._y = y
        self._origin = self._y.index[-1]
    
        self.is_initialized()
        self.validate_y()

        assert (
            len(y) >= self._seasonal_period
        ), "The data must be longer or equal to one seasonal period."

       

    def __is_fitted(self):

        assert self._y is not None
        assert self._origin is not None

    def predict(self, horizon: int = 1):

        self.__is_fitted()

        forecast_index = pd.date_range(
            start=self._origin, periods=horizon, freq=self._delta
        )

        forecast_values = []
        for h in range(1, horizon + 1):

            relative_h = h % self._seasonal_period

            if self._seasonal_period == 1:
                stepback = -self._seasonal_period 
            else:
                stepback = -self._seasonal_period  + relative_h

            forecast_values.append(self._y[stepback])

        forecast = pd.Series(data=forecast_values, index=forecast_index)

        return forecast

from itertools import cycle

import pandas as pd

from chronest import messages
from chronest.univariate.base import BaseModel


class Naive(BaseModel):
    """A naive forecasting model that simply repeats the last observed values.

    The model supports seasonality, allowing for the repetition of the last
    `seasonal_period` observed values in a cyclic manner.

    Attributes:
        _delta (pd.DateOffset): The time delta between consecutive forecasts.

        _seasonal_period (int): The number of periods to consider for the seasonality pattern.

        _y (pd.Series): The series of observed values used to fit the model.

        _origin (pd.Timestamp): The timestamp of the last observed value.

        _y_tail (pd.Series): The tail of the series, containing the last `seasonal_period` observations.
    """

    def __init__(self, delta: pd.DateOffset, seasonal_period: int = 1):
        """Initializes the Naive forecasting model.

        Args:
            delta (pd.DateOffset): The time delta between consecutive forecasts.
            seasonal_period (int): The number of periods to consider for the seasonality pattern.

        """
        super().__init__()

        if type(delta) is not pd.DateOffset:
            raise TypeError("Delta type is invalid.")
        if type(seasonal_period) is not int:
            raise TypeError("Seasonal period type is invalid.")

        self._delta = delta
        self._seasonal_period = seasonal_period

    def fit(self, y: pd.Series):
        """Fits the Naive model to the provided series.

        This method prepares the model for forecasting by storing the last
        `seasonal_period` observations and setting up the model's origin.

        Args:
            y (pd.Series): The series of observed values.
        """
        assert (
            len(y) >= self._seasonal_period
        ), "The data must be longer or equal to one seasonal period."

        self._y = y
        self._origin = self._y.index[-1]
        self._y_tail = self._y[-self._seasonal_period :]

        self.is_initialized()
        self.validate_y()

    def __is_fitted(self):
        """Checks if the model is fitted.

        Ensures that the model has been fitted with data before forecasting.

        """
        assert hasattr(self, "_y") and self._y is not None, messages.MODEL_NOT_FITTED
        assert (
            hasattr(self, "_origin") and self._origin is not None
        ), messages.MODEL_NOT_FITTED

    def predict(self, horizon: int = 1) -> pd.Series:
        """Generates forecasts for the specified horizon.

        The forecasts are generated by cyclically repeating the last `seasonal_period`
        observations for the number of periods specified by `horizon`.

        Args:
            horizon (int): The number of periods into the future to forecast.

        Returns:
            pd.Series: A series of forecasted values, indexed by the forecast dates.

        """
        self.__is_fitted()

        forecast_index = pd.date_range(
            start=self._origin + self._delta, periods=horizon, freq=self._delta
        )
        forecast_cycle = cycle(self._y_tail)
        forecast_values = [next(forecast_cycle) for _ in range(1, horizon + 1)]

        return pd.Series(data=forecast_values, index=forecast_index)
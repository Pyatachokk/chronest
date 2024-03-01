from abc import ABC

import pandas as pd

from chronest import errors, messages


class BaseModel(ABC):
    """
    Base class for all univariate forecasting models
    """

    def __init__(self) -> None:
        """
        Initialize base model class.
        """
        self._y: pd.Series = None
        self._delta: pd.DateOffset = None
        self._origin: pd.Timestamp = None

    def is_initialized(self):
        """
        Check if model was initialized
        """

        assert (
            hasattr(self, "_delta") and self._delta is not None
        ), messages.MODEL_NOT_INITIALIZED

    def validate_y(self):
        """
        Validate regularity of data index.

        """

        # Checking basic requirements on type
        if type(self._y) is not pd.Series:
            raise TypeError("Data must be pd.Series.")
        if type(self._y.index) is not pd.DatetimeIndex:
            raise TypeError("Index type must be pd.DatetimeIndex")

        self._y.sort_index(inplace=True)

        # Original index
        index: pd.DatetimeIndex = self._y.index[1:]

        # Shifted index by one delta period
        shifted_index: pd.DatetimeIndex = self._y.index[:-1] + self._delta

        # If y_t != y_{t-1} + delta, the data is irregular
        wrong_dates = index[index != shifted_index].to_list()

        if wrong_dates:
            joined_wrong_dates = ", ".join(str(date) for date in wrong_dates)
            raise errors.DateError(
                f"Deltas are not equal between several dates. \
                  Check one of following: {joined_wrong_dates}"
            )

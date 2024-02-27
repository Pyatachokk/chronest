from abc import ABC

import pandas as pd

from chronest import errors


class BaseModel(ABC):

    def __init__(self) -> None:
        self._y: pd.Series = None
        self._delta: pd.DateOffset = None
        self._origin: pd.Timestamp = None

    def is_initialized(self):
        assert self._delta is not None, "The model is not initialized."

    def validate_y(self):

        # Checking basic requirements on type
        assert type(self._y) is pd.Series
        assert type(self._y.index) is pd.DatetimeIndex

        self._y.sort_index(inplace=True)

        # Original index
        index: pd.DatetimeIndex = self._y.index[1:]

        # Shifted index by one delta period
        shifted_index: pd.DatetimeIndex = self._y.index[:-1] + self._delta

        # If y_t != y_{t-1} + delta, the data is irregular
        wrong_dates = index[index != shifted_index].to_list()

        if wrong_dates:
            joined_wrong_dates = ", ".join(wrong_dates)
            raise errors.DateError(
                f"Deltas are not equal between several dates. \
                  Check one of following: {joined_wrong_dates}"
            )

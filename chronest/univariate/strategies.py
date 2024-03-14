from copy import deepcopy
from typing import Any

import pandas as pd

from chronest import messages, errors
from chronest.univariate.base import BaseModel

MIN_SAMPLES_VALID = 2


class BaseStrategy(BaseModel):
    """
    Base class for implementing forecasting strategies for cross-sectional estimators.

    Parameters
    ----------
    delta : pd.DateOffset
        The time step between consecutive time points.

    Attributes
    ----------
    _estimator : Any
        Placeholder for the forecasting model.
    """

    def __init__(self, delta) -> None:
        super().__init__(delta=delta)
        self._estimator: Any = None

    def is_initialized(self) -> None:
        """
        Check if the strategy and its components are properly initialized.
        """
        assert self._horizon is not None, messages.STRATEGY_NOT_INITIALIZED


class DirectStrategy(BaseStrategy):
    """
    Strategy that involves direct forecasting at each horizon with a separate model.

    Parameters
    ----------
    estimator : Any
        The machine learning model or statistical method used for forecasting.
    horizon : int
        The forecast horizon.
    delta : pd.DateOffset
        The time step between consecutive forecasts.

    Attributes
    ----------
    _horizon : int
        The forecast horizon.
    _estimator : Any
        The forecasting model.
    """

    def __init__(self, estimator: Any, horizon: int, delta: pd.DateOffset):
        super().__init__(delta=delta)
        self._horizon = horizon
        self._estimator = estimator
        self.validate_horizon()

    def fit(
        self,
        X: pd.DataFrame,
        target_column: str,
        feature_subsets: dict[int, list[str]] = None,
        dir_rec: bool = False
    ) -> None:
        """
        Fit the strategy using the provided dataset.

        Parameters
        ----------
        X : pd.DataFrame
            The input features dataset. The data should be prepared without one-step shift.
        target_column : str
            The name of the target column in the dataset.
        feature_subsets : dict[int, list[str]], optional
            A dictionary where keys are horizons and values are lists of column names
            indicating the subset of features to be used for the forecast at that horizon.
            All features except target_column are used for all horizons by default.
        dir_rec : bool, optional
            If True, for each horizon in the training set will be added forecasts of 
            all previous models as features. :math:`x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}`.

            Inline math example: :math:`e^{i\\pi} + 1 = 0`

            Block math example:

            .. math::
                \\int_{-\\infty}^{\\infty} e^{-x^2} dx = \\sqrt{\\pi}
        """
        if feature_subsets is None:
            feature_subsets = {}

        self.is_initialized()

        if type(X) is not pd.DataFrame: 
            raise TypeError("X should be data frame")
        
        if X.isna().any(axis=None):
            raise ValueError("Missing values are not allowed")

        if target_column not in X.columns:
            raise ValueError("Target column is not in columns of X")
        
       

        X.sort_index(inplace=True)

        # Splitting data into target and regressors
        self._y = X[target_column]
        self._X = X.drop(columns=target_column)

        self.validate_y()

        # Origin is the last timepoint of training data
        self._origin = X.index[-1]

        if self._X.empty:
            raise ValueError("No features in the resulting X")

        if not isinstance(feature_subsets, dict):
            raise TypeError("feature_subsets should be dict.")

        # If feature_subsets are defined,
        # set of features should be defined for each horizon
        if feature_subsets:
            for h in range(1, self._horizon + 1):
                if h not in feature_subsets:
                    raise ValueError(
                        "For every horizon, there should be a subset in feature_subsets"
                    )

        self._feature_subsets = feature_subsets

        if len(self._X) <= self._horizon + MIN_SAMPLES_VALID:
            raise errors.DataError(
                f"Not enough samples for estimation. \
                Should be at greater than horizon + {MIN_SAMPLES_VALID}"
                )

        # Fitting of the models
        self._fitted_estimators = []
        for h in range(1, self._horizon + 1):

            # Shifting data to create training dataset
            y_h = self._y.copy().iloc[h:]
            X_h = self._X.iloc[:-h]


            # Subsetting if there is a certain set for horizon
            if h in feature_subsets:
                X_h = X_h.loc[:, feature_subsets[h]]

            # Copying base estimator
            estimator_h = deepcopy(self._estimator)

            # Fitting estimator for horizon
            self._fitted_estimators.append(estimator_h.fit(X=X_h, y=y_h))

    def __is_fitted(self) -> None:
        """
        Private method to check if the strategy and all its models are fitted.
        """
        assert hasattr(self, "_fitted_estimators") and self._fitted_estimators, messages.STRATEGY_NOT_INITIALIZED
        assert self._y is not None, messages.STRATEGY_NOT_INITIALIZED
        assert self._X is not None, messages.STRATEGY_NOT_INITIALIZED
        assert self._origin is not None, messages.STRATEGY_NOT_INITIALIZED
        assert (
            len(self._fitted_estimators) == self._horizon
        ), "The number of estimators is not equal to the horizon."

    def predict(self) -> pd.Series:
        """
        Generate forecasts using the fitted models.

        Returns
        -------
        pd.Series
            A time series of the forecasted values indexed by the forecast horizon dates.
        """
        self.__is_fitted()

        # Generating forward time index
        forecast_index = pd.date_range(
            start=self._origin + self._delta, periods=self._horizon, freq=self._delta
        )

        forecast_values = []

        # Getting last row of train dataset
        last_point = self._X.loc[self._origin, :].to_frame().T

        for h in range(1, self._horizon + 1):
            # Predict one point for one model
            prediction_h = self._fitted_estimators[h - 1].predict(last_point)[0]
            forecast_values.append(prediction_h)

        return pd.Series(data=forecast_values, index=forecast_index)

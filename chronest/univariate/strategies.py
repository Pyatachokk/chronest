from typing import Any
import pandas as pd
from chronest import messages
from chronest.univariate.base import BaseModel
from copy import deepcopy

class BaseStrategy(BaseModel):

    def __init__(self, delta) -> None:

        super().__init__(delta=delta)

        self._estimator: Any = None
     

    def is_initialized(self):

        """
        Check if strategy is initialized
        """

        assert self._horizon is not None, messages.STRATEGY_NOT_INITIALIZED

   
class DirectStrategy(BaseStrategy):

    def __init__(self, estimator: Any, horizon: int, delta: pd.DateOffset):

        super().__init__(delta=delta)
            
        self._horizon = horizon
        self._estimator = estimator

        self.validate_horizon()

    def fit(self,
            X: pd.DataFrame,
            target_column: str,
            feature_subsets: dict[int, list[str]]=None):
        
        if feature_subsets is None:
            feature_subsets = {}

        self.is_initialized()

        if target_column not in X.columns:
            raise ValueError("Target column is not in columns of X")
        
        X.sort_index(inplace=True)
        
        self._y = X[target_column]
        self._X = X.drop(columns=target_column)
        self._origin = X.index[-1]

        if self._X.empty:
            raise ValueError("No features in the resulting X")

        if type(feature_subsets) is not dict:
            raise TypeError("feature_subsets should be dict.")

        if feature_subsets:
            for h in range(1, self._horizon+1):
                if h not in feature_subsets:
                    raise ValueError(
                        "For every horizon should be subset in feature_subsets"
                        )
                
        self._feature_subsets = feature_subsets
        
        self._fitted_estimators = []
        for h in range(1, self._horizon+1):
            
            # Shifting target to one step at each horizon
            y_h = self._y.copy().iloc[h:]
            
            X_h = self._X.iloc[:-h]

            if h in feature_subsets:
                X_h = X_h.loc[:, self._feature_subsets]


            estimator_h = deepcopy(self._estimator)

            self._fitted_estimators.append(
                estimator_h.fit(X=X_h, y=y_h)
            )

    def __is_fitted(self):

        """
        Check if strategy is fitted
        """

        assert self._fitted_estimators, messages.STRATEGY_NOT_INITIALIZED
        assert self._y is not None, messages.STRATEGY_NOT_INITIALIZED
        assert self._X is not None, messages.STRATEGY_NOT_INITIALIZED
        assert self._origin is not None, messages.STRATEGY_NOT_INITIALIZED

        assert len(self._fitted_estimators) == self._horizon, \
            "The number of estimators is not equal to horizon"



    def predict(self):

        self.__is_fitted()

        forecast_index = pd.date_range(
            start=self._origin + self._delta, periods=self._horizon, freq=self._delta
        )

        forecast_values = []

        last_point = self._X.loc[(self._origin,),:]

        for h in range(1, self._horizon+1):
            prediction_h = self._fitted_estimators[h-1].predict(last_point)[0]
            forecast_values.append(prediction_h)


        return pd.Series(data=forecast_values, index=forecast_index)

from typing import Any
import pandas as pd
from chronest import messages

class BaseStrategy:

    def __init__(self) -> None:
        self._horizon: int = None
        self._estimator: Any = None

    def validate(self):

        if type(self._horizon) is not int:
            raise TypeError("Horizon should be integer")
        
        if self._horizon <= 0:
            raise ValueError("Horizon should be positive integer")


    def is_initialized(self):

        """
        Check if strategy is initialized
        """

        assert self._horizon is not None, messages.STRATEGY_NOT_INITIALIZED


class DirectStrategy(BaseStrategy):

    def __init__(self, estimator: Any, horizon: int):

        self.__init__()
            
        self._horizon = horizon
        self._estimator = estimator
        self._estimators = []

    def fit(self,
            X: pd.DataFrame,
            target_column: str,
            feature_subsets: dict[int, list[str]]=None):
        
        if target_column not in X.columns:
            raise ValueError("Target column is not in columns of X")
        
        X.sort_index(inplace=True)
        
        self._y = X[target_column]
        self._X = X.drop(columns=target_column)

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
                

        
            






class DirRecStrategy:

    pass
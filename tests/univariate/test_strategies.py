import pytest
from datetime import timedelta
import pandas as pd
from pandas.tseries.offsets import DateOffset
from chronest import errors
from chronest.univariate.strategies import DirectStrategy  # Update this import based on your actual module structure

# Utility functions or fixtures can go here
@pytest.fixture
def valid_dataframe():
    """Generate a valid DataFrame for testing."""
    dates = pd.date_range('2022-01-01', periods=5, freq='D')
    data = {
        "feature_1": [1, 2, 3, 4, 5],
        "target": [5, 4, 3, 2, 1]
    }
    return pd.DataFrame(data, index=dates)

@pytest.fixture
def delta():
    """Provide a default DateOffset for testing."""
    return DateOffset(days=1)

@pytest.fixture
def estimator():
    """Mock estimator for testing."""
    class MockEstimator:
        def fit(self, X, y):
            return self
        def predict(self, X):
            return [42]  # Arbitrary prediction
    return MockEstimator()

# Test initialization and parameter validation
def test_initialization(delta, estimator):
    strategy = DirectStrategy(estimator=estimator, horizon=3, delta=delta)
    assert strategy._delta == delta
    assert strategy._horizon == 3
    assert strategy._estimator == estimator

# Test for incorrect horizon values
@pytest.mark.parametrize("horizon", [0, -1, 1.5, "two"])
def test_invalid_horizon(delta, estimator, horizon):
    with pytest.raises((TypeError, ValueError)):
        DirectStrategy(estimator=estimator, horizon=horizon, delta=delta)

# Test fitting process
def test_fit_with_valid_data(valid_dataframe, delta, estimator):
    strategy = DirectStrategy(estimator=estimator, horizon=2, delta=delta)
    strategy.fit(X=valid_dataframe, target_column="target")
    assert strategy._fitted_estimators is not None
    assert len(strategy._fitted_estimators) == 2

def test_fit_with_invalid_horizon(valid_dataframe, delta, estimator):
    strategy = DirectStrategy(estimator=estimator, horizon=3, delta=delta)
    with pytest.raises(errors.DataError):
        strategy.fit(X=valid_dataframe, target_column="target")

def test_fit_with_invalid_data(delta, estimator):
    empty_df = ["test", ]
    strategy = DirectStrategy(estimator=estimator, horizon=3, delta=delta)
    with pytest.raises(TypeError):
        strategy.fit(X=empty_df, target_column="target")

# Test fit with various invalid inputs
def test_fit_with_invalid_data(delta, estimator):
    empty_df = pd.DataFrame()
    strategy = DirectStrategy(estimator=estimator, horizon=3, delta=delta)
    with pytest.raises(ValueError):
        strategy.fit(X=empty_df, target_column="target")

def test_fit_missing_target_column(valid_dataframe, delta, estimator):
    strategy = DirectStrategy(estimator=estimator, horizon=3, delta=delta)
    with pytest.raises(ValueError):
        strategy.fit(X=valid_dataframe, target_column="non_existent_column")

def test_fit_with_feature_subsets_missing_horizon(valid_dataframe, delta, estimator):
    strategy = DirectStrategy(estimator=estimator, horizon=3, delta=delta)
    with pytest.raises(ValueError):
        strategy.fit(X=valid_dataframe, target_column="target", feature_subsets={1: ["feature_1"]})

# Test if the strategy is fitted before prediction
def test_predict_before_fit(delta, estimator):
    strategy = DirectStrategy(estimator=estimator, horizon=3, delta=delta)
    with pytest.raises(AssertionError):
        strategy.predict()

# Test the predict method's output format
def test_predict_output_format(valid_dataframe, delta, estimator):
    strategy = DirectStrategy(estimator=estimator, horizon=2, delta=delta)
    strategy.fit(X=valid_dataframe, target_column="target")
    predictions = strategy.predict()
    assert isinstance(predictions, pd.Series)
    assert len(predictions) == 2  # Matches the horizon

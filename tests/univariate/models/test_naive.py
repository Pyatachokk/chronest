import pytest
import pandas as pd
from datetime import timedelta
from chronest.univariate.models.naive import Naive  # Adjust the import path according to your project structure
from chronest import errors

@pytest.fixture
def sample_series():
    dates = pd.date_range('2022-01-01', periods=5, freq='D')
    values = [1, 2, 3, 4, 5]
    return pd.Series(values, index=dates)

def test_initialization():
    delta = pd.DateOffset(days=1)
    model = Naive(delta=delta, seasonal_period=1)
    assert model._delta == delta
    assert model._seasonal_period == 1

    with pytest.raises(AssertionError):
        Naive(delta='1D', seasonal_period=1) 


def test_fit_with_valid_data(sample_series):
    model = Naive(delta=pd.DateOffset(days=1))
    model.fit(sample_series)
    assert model._y is not None
    assert model._origin == sample_series.index[-1]

def test_fit_with_irregular_intervals(sample_series):
    sample_series.index = sample_series.index.map(lambda x: x + pd.Timedelta(minutes=5))  # Making intervals irregular
    model = Naive(delta=pd.DateOffset(days=1))
    with pytest.raises(errors.DateError):
        model.fit(sample_series)

def test_fit_with_insufficient_length(sample_series):
    model = Naive(delta=pd.DateOffset(days=1), seasonal_period=10)
    with pytest.raises(AssertionError):
        model.fit(sample_series[:2])  # Insufficient length


def test_predict_single_step(sample_series):
    model = Naive(delta=pd.DateOffset(days=1), seasonal_period=1)
    model.fit(sample_series)
    forecast = model.predict(horizon=1)
    assert len(forecast) == 1
    assert forecast.iloc[0] == sample_series.iloc[-1]

def test_predict_multiple_steps(sample_series):
    model = Naive(delta=pd.DateOffset(days=1), seasonal_period=5)
    model.fit(sample_series)
    forecast = model.predict(horizon=3)
    assert len(forecast) == 3
    assert forecast.iloc[0] == sample_series.iloc[-1]

def test_predict_without_fitting():
    model = Naive(delta=pd.DateOffset(days=1), seasonal_period=1)
    with pytest.raises(AssertionError):
        model.predict(horizon=1)

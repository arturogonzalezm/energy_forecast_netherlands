"""
Unit tests for the ARIMAModel class.
Tests ARIMA model fitting, diagnostics, and forecasting functionality.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera

from src.models.arima_model import ARIMAModel


@pytest.fixture
def simple_time_series():
    """Create a simple time series for testing."""
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=100, freq='h')
    # Create a simple trending series with some noise
    trend = np.linspace(45, 55, 100)
    noise = np.random.normal(0, 2, 100)
    values = trend + noise
    return pd.Series(values, index=dates, name='price')


@pytest.fixture
def stationary_series():
    """Create a stationary time series for testing."""
    np.random.seed(123)
    # AR(1) process: y_t = 0.5 * y_{t-1} + epsilon_t
    n = 100
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.5 * y[t - 1] + np.random.normal(0, 1)

    dates = pd.date_range('2025-01-01', periods=n, freq='h')
    return pd.Series(y, index=dates, name='stationary_price')


@pytest.fixture
def white_noise_series():
    """Create a white noise series for testing."""
    np.random.seed(456)
    dates = pd.date_range('2025-01-01', periods=50, freq='h')
    values = np.random.normal(50, 5, 50)
    return pd.Series(values, index=dates, name='noise')


@pytest.fixture
def minimal_series():
    """Create a minimal series for edge case testing."""
    dates = pd.date_range('2025-01-01', periods=10, freq='h')
    values = [45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
    return pd.Series(values, index=dates, name='minimal')


class TestARIMAModel:
    """Test suite for ARIMAModel class."""

    def test_initialization_default_order(self, simple_time_series):
        """Test ARIMAModel initialization with default parameters."""
        model = ARIMAModel(simple_time_series)

        assert model.series.equals(simple_time_series)
        assert model.order == (1, 1, 1)
        assert model.results is None

    def test_initialization_custom_order(self, simple_time_series):
        """Test ARIMAModel initialization with custom order."""
        custom_order = (2, 1, 2)
        model = ARIMAModel(simple_time_series, order=custom_order)

        assert model.series.equals(simple_time_series)
        assert model.order == custom_order
        assert model.results is None

    def test_initialization_with_different_series_types(self):
        """Test initialization with different pandas Series configurations."""
        # Series without index
        series_no_index = pd.Series([1, 2, 3, 4, 5])
        model = ARIMAModel(series_no_index)
        assert len(model.series) == 5

        # Series with string name
        series_named = pd.Series([10, 20, 30], name="test_series")
        model = ARIMAModel(series_named)
        assert model.series.name == "test_series"

    def test_fit_successful(self, stationary_series):
        """Test successful model fitting."""
        model = ARIMAModel(stationary_series, order=(1, 0, 1))
        results = model.fit()

        # Check that results are returned and stored
        assert results is not None
        assert model.results is not None
        assert model.results == results

        # Check that results have expected attributes
        assert hasattr(results, 'aic')
        assert hasattr(results, 'bic')
        assert hasattr(results, 'pvalues')
        assert hasattr(results, 'resid')

    def test_fit_with_different_orders(self, simple_time_series):
        """Test fitting with various ARIMA orders."""
        orders_to_test = [
            (1, 1, 1),
            (2, 1, 0),
            (0, 1, 1),
            (1, 0, 0),  # AR(1)
            (0, 0, 1),  # MA(1)
        ]

        for order in orders_to_test:
            model = ARIMAModel(simple_time_series, order=order)
            results = model.fit()
            assert results is not None
            assert results.aic is not None
            assert results.bic is not None

    def test_fit_convergence_issues(self):
        """Test handling of convergence issues during fitting."""
        # Create a problematic series (constant values)
        problematic_series = pd.Series([50.0] * 20)
        model = ARIMAModel(problematic_series, order=(2, 1, 2))

        # This might raise warnings or convergence issues
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                results = model.fit()
                # If it succeeds, check basic properties
                assert results is not None
            except Exception as e:
                # Some orders might not work with constant series
                assert isinstance(e, (ValueError, np.linalg.LinAlgError))

    def test_diagnostics_after_fit(self, stationary_series):
        """Test diagnostics method after successful fitting."""
        model = ARIMAModel(stationary_series, order=(1, 0, 1))
        model.fit()

        diagnostics = model.diagnostics()

        # Check all expected keys are present
        expected_keys = ['aic', 'bic', 'p_ar_l1', 'p_ma_l1', 'lb_pvalue', 'jb_pvalue']
        for key in expected_keys:
            assert key in diagnostics

        # Check data types and ranges
        assert isinstance(diagnostics['aic'], (int, float))
        assert isinstance(diagnostics['bic'], (int, float))
        assert isinstance(diagnostics['lb_pvalue'], (int, float))
        assert isinstance(diagnostics['jb_pvalue'], (int, float))

        # P-values should be between 0 and 1 (or NaN)
        for pval_key in ['p_ar_l1', 'p_ma_l1', 'lb_pvalue', 'jb_pvalue']:
            pval = diagnostics[pval_key]
            if not np.isnan(pval):
                assert 0 <= pval <= 1

    def test_diagnostics_without_fit_raises_error(self, simple_time_series):
        """Test that diagnostics raises error when called before fitting."""
        model = ARIMAModel(simple_time_series)

        with pytest.raises(AttributeError):
            model.diagnostics()

    def test_diagnostics_parameter_availability(self, simple_time_series):
        """Test diagnostics with different model orders to check parameter availability."""
        # Test AR(1) model - should have ar.L1 parameter
        model_ar = ARIMAModel(simple_time_series, order=(1, 1, 0))
        model_ar.fit()
        diag_ar = model_ar.diagnostics()
        # ar.L1 should exist, ma.L1 should be NaN
        assert not np.isnan(diag_ar['p_ar_l1'])
        assert np.isnan(diag_ar['p_ma_l1'])

        # Test MA(1) model - should have ma.L1 parameter
        model_ma = ARIMAModel(simple_time_series, order=(0, 1, 1))
        model_ma.fit()
        diag_ma = model_ma.diagnostics()
        # ma.L1 should exist, ar.L1 should be NaN
        assert not np.isnan(diag_ma['p_ma_l1'])
        assert np.isnan(diag_ma['p_ar_l1'])

    def test_forecast_default_steps(self, stationary_series):
        """Test forecasting with default number of steps."""
        model = ARIMAModel(stationary_series)
        model.fit()

        forecast_df = model.forecast()

        # Check forecast DataFrame properties
        assert isinstance(forecast_df, pd.DataFrame)
        assert len(forecast_df) == 24  # Default steps

        # Check expected columns
        expected_columns = ['mean', 'mean_se', 'mean_ci_lower', 'mean_ci_upper']
        for col in expected_columns:
            assert col in forecast_df.columns

        # Check that all values are numeric
        for col in forecast_df.columns:
            assert pd.api.types.is_numeric_dtype(forecast_df[col])

    def test_forecast_custom_steps(self, stationary_series):
        """Test forecasting with custom number of steps."""
        model = ARIMAModel(stationary_series)
        model.fit()

        custom_steps = 12
        forecast_df = model.forecast(steps=custom_steps)

        assert len(forecast_df) == custom_steps
        assert isinstance(forecast_df, pd.DataFrame)

    def test_forecast_without_fit_raises_error(self, simple_time_series):
        """Test that forecast raises error when called before fitting."""
        model = ARIMAModel(simple_time_series)

        with pytest.raises(AttributeError):
            model.forecast()

    def test_forecast_confidence_intervals(self, stationary_series):
        """Test that confidence intervals are properly ordered."""
        model = ARIMAModel(stationary_series)
        model.fit()

        forecast_df = model.forecast(steps=10)

        # Lower bound should be less than mean, mean less than upper bound
        assert all(forecast_df['mean_ci_lower'] <= forecast_df['mean'])
        assert all(forecast_df['mean'] <= forecast_df['mean_ci_upper'])

        # Standard errors should be positive
        assert all(forecast_df['mean_se'] > 0)

    def test_edge_case_minimal_data(self, minimal_series):
        """Test behavior with minimal amount of data."""
        model = ARIMAModel(minimal_series, order=(1, 0, 0))  # Simple AR(1)

        # Should be able to fit even with minimal data
        results = model.fit()
        assert results is not None

        # Should be able to get diagnostics
        diagnostics = model.diagnostics()
        assert 'aic' in diagnostics

        # Should be able to forecast
        forecast_df = model.forecast(steps=5)
        assert len(forecast_df) == 5

    def test_edge_case_single_value_differencing(self):
        """Test handling of series that become very short after differencing."""
        # Series with only a few values that will be consumed by differencing
        short_series = pd.Series([50, 51, 52])
        model = ARIMAModel(short_series, order=(1, 2, 1))  # d=2 will consume 2 observations

        # This should either work or raise a meaningful error
        try:
            model.fit()
            # If it works, that's fine too - just verify we get a result
            assert model.results is not None
        except (ValueError, IndexError, TypeError) as e:
            # Various error messages can occur with insufficient data
            error_msg = str(e).lower()
            expected_indicators = [
                "observations", "insufficient", "array", "indices",
                "dimension", "empty", "short", "few", "length"
            ]
            # Check that the error message contains at least one indicator of data insufficiency
            assert any(indicator in error_msg for indicator in expected_indicators), \
                f"Unexpected error message: {str(e)}"

    def test_warnings_suppression(self, simple_time_series):
        """Test that statsmodels frequency warnings are suppressed."""
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")

            model = ARIMAModel(simple_time_series)
            model.fit()

            # Check that no frequency warnings were raised
            frequency_warnings = [w for w in warning_list
                                  if "frequency" in str(w.message).lower()]
            assert len(frequency_warnings) == 0

    @pytest.mark.parametrize("order", [
        (1, 1, 1),
        (2, 1, 0),
        (0, 1, 2),
        (1, 0, 1),
        (2, 0, 2),
    ])
    def test_various_arima_orders(self, simple_time_series, order):
        """Parametrized test for various ARIMA orders."""
        model = ARIMAModel(simple_time_series, order=order)
        results = model.fit()

        assert results is not None
        assert model.results is not None

        # Should be able to get diagnostics
        diagnostics = model.diagnostics()
        assert isinstance(diagnostics, dict)

        # Should be able to forecast
        forecast_df = model.forecast(steps=6)
        assert len(forecast_df) == 6

    def test_series_with_missing_values(self):
        """Test handling of series with NaN values."""
        series_with_nan = pd.Series([1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10])
        model = ARIMAModel(series_with_nan)

        # ARIMA should handle NaN values appropriately
        try:
            results = model.fit()
            # If successful, verify basic properties
            assert results is not None
        except ValueError as e:
            # Some ARIMA implementations don't handle NaN well
            assert "nan" in str(e).lower() or "missing" in str(e).lower()

    def test_large_forecast_horizon(self, stationary_series):
        """Test forecasting with a large number of steps."""
        model = ARIMAModel(stationary_series)
        model.fit()

        large_steps = 100
        forecast_df = model.forecast(steps=large_steps)

        assert len(forecast_df) == large_steps

        # For large horizons, confidence intervals should generally widen
        ci_width_start = (forecast_df['mean_ci_upper'].iloc[0] -
                          forecast_df['mean_ci_lower'].iloc[0])
        ci_width_end = (forecast_df['mean_ci_upper'].iloc[-1] -
                        forecast_df['mean_ci_lower'].iloc[-1])

        # Generally, uncertainty should increase with forecast horizon
        assert ci_width_end >= ci_width_start

    def test_model_refit_capability(self, simple_time_series):
        """Test that model can be refit with new data."""
        model = ARIMAModel(simple_time_series)

        # First fit
        results1 = model.fit()
        aic1 = results1.aic

        # Create new series and refit
        new_series = simple_time_series * 1.1  # Scale the series
        model.series = new_series
        results2 = model.fit()
        aic2 = results2.aic

        # Results should be different
        assert results1 != results2
        assert aic1 != aic2
        assert model.results == results2

    def test_diagnostics_statistical_validity(self, stationary_series):
        """Test that diagnostic statistics are within expected ranges."""
        model = ARIMAModel(stationary_series, order=(1, 0, 1))
        model.fit()

        diagnostics = model.diagnostics()

        # AIC and BIC should be real numbers
        assert np.isfinite(diagnostics['aic'])
        assert np.isfinite(diagnostics['bic'])

        # For most reasonable models, AIC and BIC should be positive
        # (though this isn't always guaranteed)
        assert isinstance(diagnostics['aic'], (int, float))
        assert isinstance(diagnostics['bic'], (int, float))

        # P-values should be valid probabilities or NaN
        for pval_key in ['p_ar_l1', 'p_ma_l1', 'lb_pvalue', 'jb_pvalue']:
            pval = diagnostics[pval_key]
            assert np.isnan(pval) or (0 <= pval <= 1)

    def test_error_handling_invalid_order(self, simple_time_series):
        """Test error handling for invalid ARIMA orders."""
        # Negative orders should raise errors
        with pytest.raises((ValueError, TypeError)):
            model = ARIMAModel(simple_time_series, order=(-1, 1, 1))
            model.fit()

        # Non-integer orders should raise errors
        with pytest.raises((ValueError, TypeError)):
            model = ARIMAModel(simple_time_series, order=(1.5, 1, 1))
            model.fit()

    def test_memory_efficiency_large_series(self):
        """Test that the model handles reasonably large series efficiently."""
        # Create a larger series
        np.random.seed(789)
        large_series = pd.Series(np.random.normal(50, 10, 1000))

        model = ARIMAModel(large_series, order=(1, 1, 1))

        # Should complete in reasonable time and not consume excessive memory
        import time
        start_time = time.time()
        results = model.fit()
        fit_time = time.time() - start_time

        assert results is not None
        assert fit_time < 30  # Should complete within 30 seconds

        # Should be able to get diagnostics and forecast
        diagnostics = model.diagnostics()
        forecast_df = model.forecast(steps=24)

        assert len(forecast_df) == 24
        assert isinstance(diagnostics, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

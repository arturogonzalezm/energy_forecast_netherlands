"""
Unit tests for the ElectricityForecastPipeline class.
Tests the complete pipeline orchestration with mocked dependencies.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from io import StringIO
import sys

from src.pipeline import ElectricityForecastPipeline


@pytest.fixture
def mock_spark_dataframe():
    """Create a mock Spark DataFrame for testing."""
    mock_df = Mock()

    # Mock the select method to return another mock DataFrame
    mock_selected_df = Mock()
    mock_df.select.return_value = mock_selected_df

    # Create sample pandas DataFrame for toPandas() conversion
    sample_data = {
        'timestamp': pd.date_range('2025-07-01 10:00:00', periods=100, freq='h'),
        'price': np.random.uniform(40, 60, 100)
    }
    sample_pandas_df = pd.DataFrame(sample_data)
    mock_selected_df.toPandas.return_value = sample_pandas_df

    return mock_df


@pytest.fixture
def mock_arima_results():
    """Create a mock ARIMA results object."""
    mock_results = Mock()
    mock_results.summary.return_value = "Mock ARIMA Summary\nAIC: 123.45\nBIC: 130.67"
    mock_results.aic = 123.45
    mock_results.bic = 130.67
    return mock_results


@pytest.fixture
def mock_diagnostics():
    """Create mock diagnostics dictionary."""
    return {
        'aic': 123.45,
        'bic': 130.67,
        'p_ar_l1': 0.023,
        'p_ma_l1': 0.045,
        'lb_pvalue': 0.567,
        'jb_pvalue': 0.234
    }


@pytest.fixture
def mock_forecast_df():
    """Create a mock forecast DataFrame."""
    forecast_data = {
        'mean': np.random.uniform(45, 55, 24),
        'mean_se': np.random.uniform(1, 3, 24),
        'mean_ci_lower': np.random.uniform(40, 50, 24),
        'mean_ci_upper': np.random.uniform(50, 60, 24)
    }
    return pd.DataFrame(forecast_data)


@pytest.fixture
def sample_csv_path(tmp_path):
    """Create a sample CSV file for testing."""
    content = (
        "Title line to skip\n"
        "Header1,Header2,Header3,Header4,Header5\n"
        "SubH1,SubH2,SubH3,SubH4,SubH5\n"
        "2025-07-01T10:00+02:00,100.0,200.0,300.0,50.0\n"
        "2025-07-01T11:00+02:00,110.0,210.0,310.0,52.0\n"
        "2025-07-01T12:00+02:00,105.0,205.0,305.0,51.0\n"
    )
    file_path = tmp_path / "test_energy_data.csv"
    file_path.write_text(content)
    return str(file_path)


class TestElectricityForecastPipeline:
    """Test suite for ElectricityForecastPipeline class."""

    def test_initialization(self, sample_csv_path):
        """Test pipeline initialization."""
        pipeline = ElectricityForecastPipeline(sample_csv_path)
        assert pipeline.data_path == sample_csv_path

    @patch('src.pipeline.ARIMAModel')
    @patch('src.pipeline.FeatureEngineer')
    @patch('src.pipeline.DataLoader')
    def test_run_complete_pipeline_success(
            self,
            mock_data_loader_class,
            mock_feature_engineer_class,
            mock_arima_model_class,
            mock_spark_dataframe,
            mock_arima_results,
            mock_diagnostics,
            mock_forecast_df,
            sample_csv_path
    ):
        """Test successful execution of the complete pipeline."""
        # Setup mocks
        mock_loader = Mock()
        mock_loader.load.return_value = mock_spark_dataframe
        mock_data_loader_class.return_value = mock_loader

        mock_fe = Mock()
        mock_fe.transform.return_value = mock_spark_dataframe
        mock_feature_engineer_class.return_value = mock_fe

        mock_model = Mock()
        mock_model.fit.return_value = mock_arima_results
        mock_model.diagnostics.return_value = mock_diagnostics
        mock_model.forecast.return_value = mock_forecast_df
        mock_arima_model_class.return_value = mock_model

        # Execute pipeline
        pipeline = ElectricityForecastPipeline(sample_csv_path)

        # Capture stdout to test print statements
        captured_output = StringIO()
        sys.stdout = captured_output

        try:
            results, metrics, forecast = pipeline.run()
        finally:
            sys.stdout = sys.__stdout__

        # Verify return values
        assert results == mock_arima_results
        assert metrics == mock_diagnostics
        assert forecast.equals(mock_forecast_df)

        # Verify method calls
        mock_data_loader_class.assert_called_once_with(sample_csv_path)
        mock_loader.load.assert_called_once()
        mock_feature_engineer_class.assert_called_once_with(mock_spark_dataframe)
        mock_fe.transform.assert_called_once()
        mock_spark_dataframe.select.assert_called_once_with("timestamp", "price")
        mock_arima_model_class.assert_called_once()
        mock_model.fit.assert_called_once()
        mock_model.diagnostics.assert_called_once()
        mock_model.forecast.assert_called_once()

        # Verify output contains expected content
        output = captured_output.getvalue()
        assert "Model Quality Metrics:" in output
        assert "AIC: 123.45" in output
        assert "BIC: 130.67" in output
        assert "Forecast for the next 24 hours:" in output

    @patch('src.pipeline.DataLoader')
    def test_data_loader_integration(self, mock_data_loader_class, sample_csv_path):
        """Test DataLoader integration and proper path passing."""
        mock_loader = Mock()
        mock_data_loader_class.return_value = mock_loader

        # Mock the rest of the pipeline to focus on DataLoader
        with patch('src.pipeline.FeatureEngineer'), \
                patch('src.pipeline.ARIMAModel'):

            pipeline = ElectricityForecastPipeline(sample_csv_path)

            # This will fail at feature engineering, but we can test DataLoader call
            try:
                pipeline.run()
            except:
                pass  # Expected to fail due to incomplete mocking

            # Verify DataLoader was called with correct path
            mock_data_loader_class.assert_called_once_with(sample_csv_path)
            mock_loader.load.assert_called_once()

    @patch('src.pipeline.ARIMAModel')
    @patch('src.pipeline.FeatureEngineer')
    @patch('src.pipeline.DataLoader')
    def test_feature_engineer_integration(
            self,
            mock_data_loader_class,
            mock_feature_engineer_class,
            mock_arima_model_class,
            mock_spark_dataframe,
            sample_csv_path
    ):
        """Test FeatureEngineer integration and DataFrame flow."""
        # Setup DataLoader mock
        mock_loader = Mock()
        mock_loader.load.return_value = mock_spark_dataframe
        mock_data_loader_class.return_value = mock_loader

        # Setup FeatureEngineer mock
        mock_fe = Mock()
        transformed_df = Mock()
        mock_fe.transform.return_value = transformed_df
        mock_feature_engineer_class.return_value = mock_fe

        # Mock the select chain for transformed DataFrame
        mock_selected = Mock()
        transformed_df.select.return_value = mock_selected
        sample_pandas = pd.DataFrame({
            'timestamp': pd.date_range('2025-07-01', periods=10, freq='h'),
            'price': np.random.uniform(40, 60, 10)
        })
        mock_selected.toPandas.return_value = sample_pandas

        # Mock ARIMA model to prevent further execution
        mock_model = Mock()
        mock_model.fit.return_value = Mock()
        mock_model.diagnostics.return_value = {
            'aic': 123.45, 'bic': 130.67, 'p_ar_l1': 0.023,
            'p_ma_l1': 0.045, 'lb_pvalue': 0.567, 'jb_pvalue': 0.234
        }
        mock_model.forecast.return_value = pd.DataFrame({'mean': [50.0]})
        mock_arima_model_class.return_value = mock_model

        pipeline = ElectricityForecastPipeline(sample_csv_path)

        try:
            pipeline.run()
        except:
            pass  # May fail at ARIMA stage, but we're testing up to feature engineering

        # Verify FeatureEngineer was called with raw DataFrame
        mock_feature_engineer_class.assert_called_once_with(mock_spark_dataframe)
        mock_fe.transform.assert_called_once()

        # Verify transformed DataFrame was used for selection
        transformed_df.select.assert_called_once_with("timestamp", "price")

    @patch('src.pipeline.ARIMAModel')
    @patch('src.pipeline.FeatureEngineer')
    @patch('src.pipeline.DataLoader')
    def test_pandas_conversion_and_indexing(
            self,
            mock_data_loader_class,
            mock_feature_engineer_class,
            mock_arima_model_class,
            sample_csv_path
    ):
        """Test pandas conversion and timestamp indexing."""
        # Create realistic test data
        test_timestamps = pd.date_range('2025-07-01 10:00:00', periods=50, freq='h')
        test_prices = np.random.uniform(40, 60, 50)
        test_pandas_df = pd.DataFrame({
            'timestamp': test_timestamps,
            'price': test_prices
        })

        # Setup complete mock chain
        mock_loader = Mock()
        mock_df_raw = Mock()
        mock_loader.load.return_value = mock_df_raw
        mock_data_loader_class.return_value = mock_loader

        mock_fe = Mock()
        mock_df_fe = Mock()
        mock_fe.transform.return_value = mock_df_fe
        mock_feature_engineer_class.return_value = mock_fe

        mock_selected_df = Mock()
        mock_df_fe.select.return_value = mock_selected_df
        mock_selected_df.toPandas.return_value = test_pandas_df

        # Capture the series passed to ARIMAModel
        captured_series = None

        def capture_series(series):
            nonlocal captured_series
            captured_series = series
            mock_model = Mock()
            mock_model.fit.return_value = Mock()
            # Provide proper diagnostics with all expected keys
            mock_model.diagnostics.return_value = {
                'aic': 123.45,
                'bic': 130.67,
                'p_ar_l1': 0.023,
                'p_ma_l1': 0.045,
                'lb_pvalue': 0.567,
                'jb_pvalue': 0.234
            }
            mock_model.forecast.return_value = pd.DataFrame({
                'mean': [50.0, 51.0, 52.0],
                'mean_se': [1.0, 1.1, 1.2]
            })
            return mock_model

        mock_arima_model_class.side_effect = capture_series

        pipeline = ElectricityForecastPipeline(sample_csv_path)

        # Capture output to prevent cluttering test output
        captured_output = StringIO()
        sys.stdout = captured_output
        try:
            pipeline.run()
        finally:
            sys.stdout = sys.__stdout__

        # Verify pandas conversion and indexing
        assert captured_series is not None
        assert isinstance(captured_series, pd.Series)
        assert captured_series.name == 'price'
        assert len(captured_series) == 50

        # Verify timestamp was set as index
        assert isinstance(captured_series.index, pd.DatetimeIndex)

    @patch('src.pipeline.ARIMAModel')
    @patch('src.pipeline.FeatureEngineer')
    @patch('src.pipeline.DataLoader')
    def test_error_handling_data_loader_failure(
            self,
            mock_data_loader_class,
            mock_feature_engineer_class,
            mock_arima_model_class,
            sample_csv_path
    ):
        """Test error handling when DataLoader fails."""
        # Setup DataLoader to raise an exception
        mock_loader = Mock()
        mock_loader.load.side_effect = FileNotFoundError("Test file not found")
        mock_data_loader_class.return_value = mock_loader

        pipeline = ElectricityForecastPipeline(sample_csv_path)

        with pytest.raises(FileNotFoundError, match="Test file not found"):
            pipeline.run()

    @patch('src.pipeline.ARIMAModel')
    @patch('src.pipeline.FeatureEngineer')
    @patch('src.pipeline.DataLoader')
    def test_error_handling_feature_engineer_failure(
            self,
            mock_data_loader_class,
            mock_feature_engineer_class,
            mock_arima_model_class,
            mock_spark_dataframe,
            sample_csv_path
    ):
        """Test error handling when FeatureEngineer fails."""
        # Setup DataLoader to succeed
        mock_loader = Mock()
        mock_loader.load.return_value = mock_spark_dataframe
        mock_data_loader_class.return_value = mock_loader

        # Setup FeatureEngineer to fail
        mock_fe = Mock()
        mock_fe.transform.side_effect = ValueError("Feature engineering failed")
        mock_feature_engineer_class.return_value = mock_fe

        pipeline = ElectricityForecastPipeline(sample_csv_path)

        with pytest.raises(ValueError, match="Feature engineering failed"):
            pipeline.run()

    @patch('src.pipeline.ARIMAModel')
    @patch('src.pipeline.FeatureEngineer')
    @patch('src.pipeline.DataLoader')
    def test_error_handling_arima_model_failure(
            self,
            mock_data_loader_class,
            mock_feature_engineer_class,
            mock_arima_model_class,
            mock_spark_dataframe,
            sample_csv_path
    ):
        """Test error handling when ARIMA model fails."""
        # Setup successful DataLoader and FeatureEngineer
        mock_loader = Mock()
        mock_loader.load.return_value = mock_spark_dataframe
        mock_data_loader_class.return_value = mock_loader

        mock_fe = Mock()
        mock_fe.transform.return_value = mock_spark_dataframe
        mock_feature_engineer_class.return_value = mock_fe

        # Setup ARIMA model to fail during fitting
        mock_model = Mock()
        mock_model.fit.side_effect = ValueError("ARIMA fitting failed")
        mock_arima_model_class.return_value = mock_model

        pipeline = ElectricityForecastPipeline(sample_csv_path)

        with pytest.raises(ValueError, match="ARIMA fitting failed"):
            pipeline.run()

    @patch('src.pipeline.ARIMAModel')
    @patch('src.pipeline.FeatureEngineer')
    @patch('src.pipeline.DataLoader')
    def test_output_formatting(
            self,
            mock_data_loader_class,
            mock_feature_engineer_class,
            mock_arima_model_class,
            mock_spark_dataframe,
            sample_csv_path
    ):
        """Test output formatting and print statements."""
        # Setup complete successful pipeline
        mock_loader = Mock()
        mock_loader.load.return_value = mock_spark_dataframe
        mock_data_loader_class.return_value = mock_loader

        mock_fe = Mock()
        mock_fe.transform.return_value = mock_spark_dataframe
        mock_feature_engineer_class.return_value = mock_fe

        # Create detailed mock results
        mock_results = Mock()
        mock_results.summary.return_value = "Detailed ARIMA Model Summary"

        mock_diagnostics = {
            'aic': 123.456789,
            'bic': 130.123456,
            'p_ar_l1': 0.023456,
            'p_ma_l1': 0.045678,
            'lb_pvalue': 0.567890,
            'jb_pvalue': 0.234567
        }

        mock_forecast = pd.DataFrame({
            'mean': [50.1, 50.2, 50.3],
            'mean_se': [1.1, 1.2, 1.3]
        })

        mock_model = Mock()
        mock_model.fit.return_value = mock_results
        mock_model.diagnostics.return_value = mock_diagnostics
        mock_model.forecast.return_value = mock_forecast
        mock_arima_model_class.return_value = mock_model

        pipeline = ElectricityForecastPipeline(sample_csv_path)

        # Capture all output
        captured_output = StringIO()
        sys.stdout = captured_output

        try:
            pipeline.run()
        finally:
            sys.stdout = sys.__stdout__

        output = captured_output.getvalue()

        # Verify summary is printed
        assert "Detailed ARIMA Model Summary" in output

        # Verify metrics formatting
        assert "Model Quality Metrics:" in output
        assert "AIC: 123.46" in output  # Check decimal formatting
        assert "BIC: 130.12" in output
        assert "p-value AR(1): 0.023" in output  # Check 3 decimal places
        assert "MA(1): 0.046" in output
        assert "Ljung-Box p-value: 0.568" in output
        assert "Jarque-Bera p-value: 0.235" in output

        # Verify forecast section
        assert "Forecast for the next 24 hours:" in output

    def test_pipeline_with_different_file_paths(self):
        """Test pipeline initialization with various file paths."""
        test_paths = [
            "/path/to/file.csv",
            "relative/path/data.csv",
            "data.csv",
            "/very/long/path/to/energy/data/file.csv"
        ]

        for path in test_paths:
            pipeline = ElectricityForecastPipeline(path)
            assert pipeline.data_path == path

    @patch('builtins.print')
    @patch('src.pipeline.ARIMAModel')
    @patch('src.pipeline.FeatureEngineer')
    @patch('src.pipeline.DataLoader')
    def test_print_calls_verification(
            self,
            mock_data_loader_class,
            mock_feature_engineer_class,
            mock_arima_model_class,
            mock_print,
            mock_spark_dataframe,
            sample_csv_path
    ):
        """Test that all expected print statements are called."""
        # Setup successful pipeline
        mock_loader = Mock()
        mock_loader.load.return_value = mock_spark_dataframe
        mock_data_loader_class.return_value = mock_loader

        mock_fe = Mock()
        mock_fe.transform.return_value = mock_spark_dataframe
        mock_feature_engineer_class.return_value = mock_fe

        mock_results = Mock()
        mock_results.summary.return_value = "Summary"

        mock_model = Mock()
        mock_model.fit.return_value = mock_results
        mock_model.diagnostics.return_value = {
            'aic': 123.45, 'bic': 130.67, 'p_ar_l1': 0.023,
            'p_ma_l1': 0.045, 'lb_pvalue': 0.567, 'jb_pvalue': 0.234
        }
        mock_model.forecast.return_value = pd.DataFrame({'mean': [50.0]})
        mock_arima_model_class.return_value = mock_model

        pipeline = ElectricityForecastPipeline(sample_csv_path)
        pipeline.run()

        # Verify print was called multiple times
        assert mock_print.call_count >= 4  # Summary + metrics header + metrics + forecast header

    @patch('src.pipeline.ARIMAModel')
    @patch('src.pipeline.FeatureEngineer')
    @patch('src.pipeline.DataLoader')
    def test_error_handling_incomplete_diagnostics(
            self,
            mock_data_loader_class,
            mock_feature_engineer_class,
            mock_arima_model_class,
            mock_spark_dataframe,
            sample_csv_path
    ):
        """Test error handling when diagnostics are incomplete."""
        # Setup successful DataLoader and FeatureEngineer
        mock_loader = Mock()
        mock_loader.load.return_value = mock_spark_dataframe
        mock_data_loader_class.return_value = mock_loader

        mock_fe = Mock()
        mock_fe.transform.return_value = mock_spark_dataframe
        mock_feature_engineer_class.return_value = mock_fe

        # Setup ARIMA model with incomplete diagnostics
        mock_model = Mock()
        mock_model.fit.return_value = Mock()
        mock_model.diagnostics.return_value = {'aic': 123.45}  # Missing required keys
        mock_arima_model_class.return_value = mock_model

        pipeline = ElectricityForecastPipeline(sample_csv_path)

        with pytest.raises(KeyError):  # Should fail due to missing 'bic' key
            pipeline.run()

    @patch('src.pipeline.ARIMAModel')
    @patch('src.pipeline.FeatureEngineer')
    @patch('src.pipeline.DataLoader')
    def test_series_preparation_edge_cases(
            self,
            mock_data_loader_class,
            mock_feature_engineer_class,
            mock_arima_model_class,
            sample_csv_path
    ):
        """Test edge cases in series preparation for ARIMA."""
        # Test with empty DataFrame
        empty_pandas_df = pd.DataFrame({'timestamp': [], 'price': []})

        mock_loader = Mock()
        mock_df_raw = Mock()
        mock_loader.load.return_value = mock_df_raw
        mock_data_loader_class.return_value = mock_loader

        mock_fe = Mock()
        mock_df_fe = Mock()
        mock_fe.transform.return_value = mock_df_fe
        mock_feature_engineer_class.return_value = mock_fe

        mock_selected_df = Mock()
        mock_df_fe.select.return_value = mock_selected_df
        mock_selected_df.toPandas.return_value = empty_pandas_df

        # ARIMA should receive empty series
        captured_series = None

        def capture_empty_series(series):
            nonlocal captured_series
            captured_series = series
            # This would likely fail in real ARIMA, but we'll mock success
            mock_model = Mock()
            mock_model.fit.return_value = Mock()
            mock_model.diagnostics.return_value = {
                'aic': float('inf'), 'bic': float('inf'), 'p_ar_l1': float('nan'),
                'p_ma_l1': float('nan'), 'lb_pvalue': float('nan'), 'jb_pvalue': float('nan')
            }
            mock_model.forecast.return_value = pd.DataFrame()
            return mock_model

        mock_arima_model_class.side_effect = capture_empty_series

        pipeline = ElectricityForecastPipeline(sample_csv_path)

        # Capture output
        captured_output = StringIO()
        sys.stdout = captured_output
        try:
            pipeline.run()
        finally:
            sys.stdout = sys.__stdout__

        # Verify empty series was passed
        assert captured_series is not None
        assert len(captured_series) == 0
        assert captured_series.name == 'price'

    @patch('src.pipeline.ARIMAModel')
    @patch('src.pipeline.FeatureEngineer')
    @patch('src.pipeline.DataLoader')
    def test_simplified_pandas_conversion(
            self,
            mock_data_loader_class,
            mock_feature_engineer_class,
            mock_arima_model_class,
            sample_csv_path
    ):
        """Simplified test for pandas conversion without complex capture logic."""
        # Create test data
        test_timestamps = pd.date_range('2025-07-01 10:00:00', periods=10, freq='h')
        test_prices = [50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0]
        test_pandas_df = pd.DataFrame({
            'timestamp': test_timestamps,
            'price': test_prices
        })

        # Setup mocks
        mock_loader = Mock()
        mock_df_raw = Mock()
        mock_loader.load.return_value = mock_df_raw
        mock_data_loader_class.return_value = mock_loader

        mock_fe = Mock()
        mock_df_fe = Mock()
        mock_fe.transform.return_value = mock_df_fe
        mock_feature_engineer_class.return_value = mock_fe

        mock_selected_df = Mock()
        mock_df_fe.select.return_value = mock_selected_df
        mock_selected_df.toPandas.return_value = test_pandas_df

        # Setup ARIMA model with complete mocks
        mock_model = Mock()
        mock_results = Mock()
        mock_results.summary.return_value = "Test ARIMA Summary"
        mock_model.fit.return_value = mock_results
        mock_model.diagnostics.return_value = {
            'aic': 100.25,
            'bic': 105.50,
            'p_ar_l1': 0.012,
            'p_ma_l1': 0.034,
            'lb_pvalue': 0.789,
            'jb_pvalue': 0.456
        }
        mock_model.forecast.return_value = pd.DataFrame({
            'mean': [60.0, 61.0, 62.0],
            'mean_se': [1.5, 1.6, 1.7]
        })
        mock_arima_model_class.return_value = mock_model

        # Run pipeline
        pipeline = ElectricityForecastPipeline(sample_csv_path)

        # Capture output
        captured_output = StringIO()
        sys.stdout = captured_output
        try:
            results, metrics, forecast = pipeline.run()
        finally:
            sys.stdout = sys.__stdout__

        # Verify results
        assert results == mock_results
        assert metrics['aic'] == 100.25
        assert metrics['bic'] == 105.50
        assert len(forecast) == 3

        # Verify the DataFrame selection was called correctly
        mock_df_fe.select.assert_called_once_with("timestamp", "price")
        mock_selected_df.toPandas.assert_called_once()

        # Verify ARIMA model was created (we can't easily verify the exact series without capture)
        mock_arima_model_class.assert_called_once()

        # Verify output formatting
        output = captured_output.getvalue()
        assert "Test ARIMA Summary" in output
        assert "AIC: 100.25" in output
        assert "BIC: 105.50" in output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

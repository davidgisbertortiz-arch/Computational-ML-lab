"""Tests for forecasting and backtesting implementation."""

import pytest
import numpy as np
from modules._import_helper import safe_import_from

(RollingWindowBacktest, naive_forecast, moving_average_forecast,
 seasonal_naive_forecast, exponential_smoothing_forecast, LeakageGuard,
 compute_forecast_metrics) = safe_import_from(
    '04_time_series_state_space.src.forecasting',
    'RollingWindowBacktest', 'naive_forecast', 'moving_average_forecast',
    'seasonal_naive_forecast', 'exponential_smoothing_forecast', 'LeakageGuard',
    'compute_forecast_metrics'
)

set_seed = safe_import_from('00_repo_standards.src.mlphys_core', 'set_seed')


class TestRollingWindowBacktest:
    """Test rolling-window backtesting framework."""
    
    def test_split_generation(self):
        """Test train/test split generation."""
        backtest = RollingWindowBacktest(train_size=50, test_size=10, step_size=10)
        splits = backtest.get_splits(100)
        
        assert len(splits) > 0
        
        # Check first split
        train_idx, test_idx = splits[0]
        assert len(train_idx) == 50
        assert len(test_idx) == 10
        assert train_idx[-1] + 1 == test_idx[0]  # Contiguous
        
    def test_temporal_ordering(self):
        """Test that splits maintain temporal order."""
        backtest = RollingWindowBacktest(train_size=30, test_size=10, step_size=10)
        splits = backtest.get_splits(100)
        
        for train_idx, test_idx in splits:
            # Test comes after train
            assert np.max(train_idx) < np.min(test_idx)
            
    def test_run_backtesting(self):
        """Test running complete backtesting."""
        set_seed(42)
        y = np.sin(np.linspace(0, 10, 100))
        
        backtest = RollingWindowBacktest(train_size=30, test_size=10, step_size=10)
        results = backtest.run(y, naive_forecast)
        
        assert "forecasts" in results
        assert "actuals" in results
        assert "mae" in results
        assert "rmse" in results
        assert "metrics" in results
        
        assert len(results["forecasts"]) > 0
        assert len(results["mae"]) == len(results["forecasts"])
        
    def test_insufficient_data_error(self):
        """Test error handling for insufficient data."""
        y = np.array([1, 2, 3])  # Too short
        backtest = RollingWindowBacktest(train_size=50, test_size=10)
        
        with pytest.raises(ValueError, match="Not enough data"):
            backtest.run(y, naive_forecast)


class TestForecastingMethods:
    """Test forecasting baseline methods."""
    
    def test_naive_forecast(self):
        """Test naive forecasting."""
        y_train = np.array([1, 2, 3, 4, 5])
        horizon = 3
        
        y_pred = naive_forecast(y_train, horizon)
        
        assert y_pred.shape == (horizon,)
        assert np.all(y_pred == 5)  # Should repeat last value
        
    def test_moving_average_forecast(self):
        """Test moving average forecasting."""
        y_train = np.array([1, 2, 3, 4, 5])
        horizon = 3
        
        y_pred = moving_average_forecast(y_train, horizon, window=3)
        
        assert y_pred.shape == (horizon,)
        expected = np.mean([3, 4, 5])  # Last 3 values
        assert np.all(y_pred == expected)
        
    def test_seasonal_naive_forecast(self):
        """Test seasonal naive forecasting."""
        # Simple seasonal pattern
        y_train = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
        horizon = 4
        period = 3
        
        y_pred = seasonal_naive_forecast(y_train, horizon, period)
        
        assert y_pred.shape == (horizon,)
        # Should repeat last seasonal pattern
        assert y_pred[0] == 1
        assert y_pred[1] == 2
        assert y_pred[2] == 3
        assert y_pred[3] == 1
        
    def test_exponential_smoothing_forecast(self):
        """Test exponential smoothing forecasting."""
        y_train = np.array([1, 2, 3, 4, 5])
        horizon = 3
        
        y_pred = exponential_smoothing_forecast(y_train, horizon, alpha=0.5)
        
        assert y_pred.shape == (horizon,)
        # All forecasts should be same (no trend/seasonality)
        assert np.all(y_pred == y_pred[0])
        # Should be between first and last value
        assert y_pred[0] > y_train[0]
        assert y_pred[0] < y_train[-1]


class TestLeakageGuard:
    """Test temporal leakage detection utilities."""
    
    def test_validate_temporal_order_valid(self):
        """Test temporal order validation with valid splits."""
        train_idx = np.array([0, 1, 2, 3, 4])
        test_idx = np.array([5, 6, 7])
        
        is_valid = LeakageGuard.validate_temporal_order(train_idx, test_idx)
        assert is_valid
        
    def test_validate_temporal_order_invalid(self):
        """Test temporal order validation with invalid splits."""
        train_idx = np.array([0, 1, 5, 6])  # Contains future indices
        test_idx = np.array([2, 3, 4])
        
        is_valid = LeakageGuard.validate_temporal_order(train_idx, test_idx)
        assert not is_valid
        
    def test_check_feature_leakage(self):
        """Test feature leakage detection."""
        X_train = np.random.randn(100, 5)
        X_test = np.random.randn(50, 5)
        
        report = LeakageGuard.check_feature_leakage(X_train, X_test)
        
        assert "n_suspicious" in report
        assert "features" in report


class TestMetrics:
    """Test forecast evaluation metrics."""
    
    def test_compute_forecast_metrics(self):
        """Test metric computation."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
        
        metrics = compute_forecast_metrics(y_true, y_pred)
        
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "mape" in metrics
        assert "directional_accuracy" in metrics
        
        assert metrics["mae"] > 0
        assert metrics["rmse"] > 0
        
    def test_perfect_forecast_metrics(self):
        """Test metrics for perfect forecast."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = y_true.copy()
        
        metrics = compute_forecast_metrics(y_true, y_pred)
        
        assert metrics["mae"] == 0.0
        assert metrics["rmse"] == 0.0


class TestIntegration:
    """Integration tests for forecasting pipeline."""
    
    def test_full_forecasting_pipeline(self):
        """Test complete forecasting workflow."""
        set_seed(42)
        
        # Generate synthetic data
        t = np.linspace(0, 20, 200)
        y = np.sin(t) + 0.1 * t + np.random.randn(200) * 0.1
        
        # Setup backtesting
        backtest = RollingWindowBacktest(
            train_size=50,
            test_size=10,
            step_size=10
        )
        
        # Test multiple methods
        methods = [
            naive_forecast,
            lambda y_train, h: moving_average_forecast(y_train, h, window=10),
        ]
        
        for method in methods:
            results = backtest.run(y, method)
            
            # Check results structure
            assert len(results["forecasts"]) > 0
            assert results["metrics"]["mean_mae"] > 0
            assert results["metrics"]["mean_rmse"] > 0
            
    def test_reproducibility_with_seed(self):
        """Test that forecasting is reproducible with seed."""
        set_seed(42)
        t = np.linspace(0, 10, 100)
        y = np.sin(t) + np.random.randn(100) * 0.1
        
        backtest = RollingWindowBacktest(train_size=30, test_size=10)
        
        # Run 1
        set_seed(42)
        results1 = backtest.run(y, naive_forecast)
        
        # Run 2 with same seed
        set_seed(42)
        results2 = backtest.run(y, naive_forecast)
        
        # Results should be identical
        assert np.allclose(results1["mae"], results2["mae"])
        assert np.allclose(results1["rmse"], results2["rmse"])

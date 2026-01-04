"""Time series forecasting with proper backtesting and leakage prevention.

Implements rolling-window backtesting framework and baseline forecasting methods.
Critical: prevents temporal data leakage through proper train/test splits.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RollingWindowBacktest:
    """
    Rolling-window backtesting for time series models.
    
    Prevents temporal data leakage by:
    1. Never using future information in training
    2. Always maintaining temporal order
    3. Fitting scalers/transformers only on training window
    
    Args:
        train_size: Number of timesteps in training window
        test_size: Number of timesteps to forecast (horizon)
        step_size: How many steps to roll forward (default: test_size)
        
    Example:
        >>> # Setup
        >>> y = np.sin(np.linspace(0, 20, 200)) + np.random.randn(200) * 0.1
        >>> backtest = RollingWindowBacktest(train_size=50, test_size=10)
        >>> 
        >>> # Run backtesting
        >>> results = backtest.run(y, naive_forecast)
        >>> print(f"Mean MAE: {np.mean(results['mae']):.3f}")
    """
    
    train_size: int
    test_size: int
    step_size: Optional[int] = None
    min_train_size: Optional[int] = None  # For expanding window
    
    def __post_init__(self):
        """Validate parameters."""
        if self.step_size is None:
            self.step_size = self.test_size
            
        if self.min_train_size is None:
            self.min_train_size = self.train_size
            
        assert self.train_size >= self.min_train_size
        assert self.test_size > 0
        assert self.step_size > 0
        
    def get_splits(self, n: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test index splits.
        
        Args:
            n: Total length of time series
            
        Returns:
            splits: List of (train_indices, test_indices) tuples
            
        Example:
            >>> backtest = RollingWindowBacktest(train_size=50, test_size=10, step_size=10)
            >>> splits = backtest.get_splits(100)
            >>> # Split 1: train=[0:50], test=[50:60]
            >>> # Split 2: train=[10:60], test=[60:70]
            >>> # Split 3: train=[20:70], test=[70:80]
            >>> # ...
        """
        splits = []
        
        start = 0
        while start + self.train_size + self.test_size <= n:
            train_end = start + self.train_size
            test_end = train_end + self.test_size
            
            train_idx = np.arange(start, train_end)
            test_idx = np.arange(train_end, test_end)
            
            splits.append((train_idx, test_idx))
            start += self.step_size
            
        return splits
    
    def run(
        self,
        y: np.ndarray,
        forecast_fn: Callable[[np.ndarray, int], np.ndarray],
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Run rolling-window backtesting.
        
        Args:
            y: Time series (n_timesteps,)
            forecast_fn: Function (y_train, horizon) -> y_forecast
            verbose: Print progress
            
        Returns:
            results: Dictionary containing:
                - forecasts: List of forecast arrays
                - actuals: List of actual arrays
                - mae: List of MAE per window
                - rmse: List of RMSE per window
                - metrics: Aggregated metrics
        """
        splits = self.get_splits(len(y))
        
        if len(splits) == 0:
            raise ValueError(
                f"Not enough data for backtesting. Need at least "
                f"{self.train_size + self.test_size} points, got {len(y)}"
            )
        
        forecasts = []
        actuals = []
        maes = []
        rmses = []
        
        for i, (train_idx, test_idx) in enumerate(splits):
            y_train = y[train_idx]
            y_test = y[test_idx]
            
            # Forecast
            y_pred = forecast_fn(y_train, len(test_idx))
            
            # Clip forecast to test size (in case forecast_fn returns more)
            y_pred = y_pred[:len(test_idx)]
            
            # Compute metrics
            mae = np.mean(np.abs(y_test - y_pred))
            rmse = np.sqrt(np.mean((y_test - y_pred)**2))
            
            forecasts.append(y_pred)
            actuals.append(y_test)
            maes.append(mae)
            rmses.append(rmse)
            
            if verbose:
                print(f"Window {i+1}/{len(splits)}: MAE={mae:.4f}, RMSE={rmse:.4f}")
        
        # Aggregate metrics
        results = {
            "forecasts": forecasts,
            "actuals": actuals,
            "mae": np.array(maes),
            "rmse": np.array(rmses),
            "metrics": {
                "mean_mae": np.mean(maes),
                "std_mae": np.std(maes),
                "mean_rmse": np.mean(rmses),
                "std_rmse": np.std(rmses),
                "n_windows": len(splits),
            },
        }
        
        return results


def naive_forecast(y_train: np.ndarray, horizon: int) -> np.ndarray:
    """
    Naive forecast: repeat last observed value.
    
    Simple but often competitive baseline for time series.
    
    Args:
        y_train: Training time series (n,)
        horizon: Forecast horizon
        
    Returns:
        y_pred: Forecast (horizon,)
    """
    return np.full(horizon, y_train[-1])


def moving_average_forecast(
    y_train: np.ndarray,
    horizon: int,
    window: int = 10
) -> np.ndarray:
    """
    Moving average forecast: average of last k values.
    
    Args:
        y_train: Training time series (n,)
        horizon: Forecast horizon
        window: Number of recent values to average
        
    Returns:
        y_pred: Forecast (horizon,)
    """
    window = min(window, len(y_train))
    avg = np.mean(y_train[-window:])
    return np.full(horizon, avg)


def seasonal_naive_forecast(
    y_train: np.ndarray,
    horizon: int,
    period: int
) -> np.ndarray:
    """
    Seasonal naive: repeat pattern from last seasonal period.
    
    Args:
        y_train: Training time series (n,)
        horizon: Forecast horizon
        period: Seasonal period (e.g., 12 for monthly with yearly seasonality)
        
    Returns:
        y_pred: Forecast (horizon,)
    """
    if len(y_train) < period:
        # Fall back to naive
        return naive_forecast(y_train, horizon)
    
    # Repeat last seasonal pattern
    last_season = y_train[-period:]
    n_repeats = int(np.ceil(horizon / period))
    y_pred = np.tile(last_season, n_repeats)[:horizon]
    
    return y_pred


def exponential_smoothing_forecast(
    y_train: np.ndarray,
    horizon: int,
    alpha: float = 0.3
) -> np.ndarray:
    """
    Simple exponential smoothing (constant level).
    
    Level equation: ℓ_t = α y_t + (1-α) ℓ_{t-1}
    Forecast: ŷ_{t+h|t} = ℓ_t
    
    Args:
        y_train: Training time series (n,)
        horizon: Forecast horizon
        alpha: Smoothing parameter (0 < alpha < 1)
        
    Returns:
        y_pred: Forecast (horizon,)
    """
    # Initialize level with first observation
    level = y_train[0]
    
    # Update level through training data
    for y_t in y_train[1:]:
        level = alpha * y_t + (1 - alpha) * level
    
    # Forecast is constant (no trend/seasonality)
    return np.full(horizon, level)


class LeakageGuard:
    """
    Utilities to detect and prevent temporal data leakage.
    
    Common leakage sources:
    1. Fitting scalers on full dataset (including test)
    2. Computing features using future information
    3. Sorting/shuffling time series data
    4. Using future information in feature engineering
    """
    
    @staticmethod
    def validate_temporal_order(
        train_indices: np.ndarray,
        test_indices: np.ndarray
    ) -> bool:
        """
        Check that test indices come after train indices.
        
        Args:
            train_indices: Training set indices
            test_indices: Test set indices
            
        Returns:
            valid: True if temporal order is preserved
        """
        if len(train_indices) == 0 or len(test_indices) == 0:
            return True
        
        max_train = np.max(train_indices)
        min_test = np.min(test_indices)
        
        return min_test > max_train
    
    @staticmethod
    def check_feature_leakage(
        X_train: np.ndarray,
        X_test: np.ndarray,
        feature_names: Optional[List[str]] = None,
        threshold: float = 0.99
    ) -> Dict[str, Any]:
        """
        Detect potential feature leakage via correlation.
        
        If test features are highly predictable from train features,
        there may be leakage (or the features are truly deterministic).
        
        Args:
            X_train: Training features (n_train, n_features)
            X_test: Test features (n_test, n_features)
            feature_names: Feature names for reporting
            threshold: Correlation threshold for flagging
            
        Returns:
            report: Dictionary with suspicious features
        """
        if X_train.shape[1] != X_test.shape[1]:
            raise ValueError("Train and test must have same number of features")
        
        n_features = X_train.shape[1]
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        
        suspicious = []
        
        for i in range(n_features):
            # Check if test feature values highly correlate with train feature
            # (This is a heuristic, not definitive)
            train_values = X_train[:, i]
            test_values = X_test[:, i]
            
            # Check for exact duplicates
            if len(np.intersect1d(train_values, test_values)) > 0.5 * len(test_values):
                suspicious.append({
                    "feature": feature_names[i],
                    "reason": "High overlap between train and test values"
                })
        
        return {
            "n_suspicious": len(suspicious),
            "features": suspicious
        }


def compute_forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive forecast evaluation metrics.
    
    Args:
        y_true: Actual values (n,)
        y_pred: Predicted values (n,)
        
    Returns:
        metrics: Dictionary of metric values
    """
    errors = y_true - y_pred
    abs_errors = np.abs(errors)
    
    # Point forecast metrics
    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(errors**2))
    mape = np.mean(abs_errors / (np.abs(y_true) + 1e-10)) * 100
    
    # Directional accuracy (for financial time series)
    if len(y_true) > 1:
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        directional_accuracy = np.mean(true_direction == pred_direction)
    else:
        directional_accuracy = np.nan
    
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "directional_accuracy": float(directional_accuracy),
    }

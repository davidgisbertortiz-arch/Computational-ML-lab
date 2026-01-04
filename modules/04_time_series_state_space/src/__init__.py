"""Module 04: Time Series & State Space Models.

This module implements state estimation and time series forecasting methods.
"""

from .kalman import KalmanFilter
from .ekf import ExtendedKalmanFilter
from .particle_filter import ParticleFilter
from .forecasting import RollingWindowBacktest, naive_forecast, moving_average_forecast
from .hybrid_models import (
    MeasurementNetwork,
    DynamicsResidualNetwork,
    HybridEKF,
    HybridParticleFilter,
    HybridEKFConfig,
)

__all__ = [
    # Classical filters
    "KalmanFilter",
    "ExtendedKalmanFilter",
    "ParticleFilter",
    # Forecasting
    "RollingWindowBacktest",
    "naive_forecast",
    "moving_average_forecast",
    # Hybrid models
    "MeasurementNetwork",
    "DynamicsResidualNetwork",
    "HybridEKF",
    "HybridParticleFilter",
    "HybridEKFConfig",
]

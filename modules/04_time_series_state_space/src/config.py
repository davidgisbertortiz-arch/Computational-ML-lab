"""Configuration schemas for Module 04 experiments."""

from pathlib import Path
from pydantic import BaseModel, Field
from typing import Literal
from modules._import_helper import safe_import_from

ExperimentConfig = safe_import_from(
    '00_repo_standards.src.mlphys_core',
    'ExperimentConfig'
)


class TrackingConfig(ExperimentConfig):
    """Configuration for state estimation tracking experiments."""
    
    # Simulation parameters
    n_timesteps: int = Field(100, description="Number of timesteps to simulate")
    dt: float = Field(0.1, description="Time step size")
    
    # System parameters
    process_noise: float = Field(0.01, description="Process noise std dev")
    obs_noise: float = Field(0.5, description="Observation noise std dev")
    
    # Filter parameters
    filter_type: Literal["kalman", "ekf", "particle"] = Field(
        "kalman",
        description="Filter type to use"
    )
    n_particles: int = Field(1000, description="Number of particles (for PF)")
    
    # System type
    system: Literal["constant_velocity", "pendulum"] = Field(
        "constant_velocity",
        description="Dynamical system to simulate"
    )
    
    # Pendulum-specific parameters
    pendulum_length: float = Field(1.0, description="Pendulum length (m)")
    gravity: float = Field(9.81, description="Gravitational acceleration (m/s²)")
    initial_angle: float = Field(0.5, description="Initial angle (rad)")
    initial_velocity: float = Field(0.0, description="Initial angular velocity (rad/s)")
    
    # Output
    save_plots: bool = Field(True, description="Save trajectory plots")
    

class ForecastingConfig(ExperimentConfig):
    """Configuration for time series forecasting experiments."""
    
    # Data parameters
    n_points: int = Field(200, description="Total number of data points")
    noise_level: float = Field(0.1, description="Noise standard deviation")
    
    # Backtesting parameters
    train_size: int = Field(50, description="Training window size")
    test_size: int = Field(10, description="Forecast horizon")
    step_size: int = Field(10, description="Rolling step size")
    
    # Forecasting method
    method: Literal["naive", "moving_average", "exponential_smoothing", "seasonal_naive"] = Field(
        "naive",
        description="Forecasting method"
    )
    
    # Method-specific parameters
    ma_window: int = Field(10, description="Moving average window")
    es_alpha: float = Field(0.3, description="Exponential smoothing alpha")
    seasonal_period: int = Field(12, description="Seasonal period")
    
    # Output
    save_plots: bool = Field(True, description="Save forecast plots")

"""
mlphys_core: Core utilities for the Computational ML Lab.

This package provides foundational utilities for all modules:
- Configuration management
- Seeding for reproducibility
- Logging and experiment tracking
- Experiment runner base class
"""

# Python 3.12+ workaround using relative imports within package
from .config import ExperimentConfig, load_config
from .seeding import set_seed, get_rng
from .logging_utils import setup_logger, log_metrics
from .experiment import BaseExperiment

__version__ = "0.1.0"

__all__ = [
    "ExperimentConfig",
    "load_config",
    "set_seed",
    "get_rng",
    "setup_logger",
    "log_metrics",
    "BaseExperiment",
]

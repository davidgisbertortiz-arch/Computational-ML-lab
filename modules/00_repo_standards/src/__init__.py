"""Module 00: Repository Standards - Math utilities example."""

# Python 3.12+ workaround - use relative imports within package
from .core import compute_mean, compute_variance, gradient_descent
from .utils import set_seed, load_config, log_experiment

__all__ = [
    "compute_mean",
    "compute_variance",
    "gradient_descent",
    "set_seed",
    "load_config",
    "log_experiment",
]

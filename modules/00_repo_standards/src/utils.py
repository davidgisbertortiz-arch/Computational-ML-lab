"""Utility functions for configuration, logging, and reproducibility."""

import yaml
import numpy as np
import random
from pathlib import Path
from typing import Any


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Sets seeds for numpy and Python's random module.
    
    Args:
        seed: Random seed (integer)
        
    Example:
        >>> set_seed(42)
        >>> np.random.randn(1)
        array([0.49671415])
    """
    np.random.seed(seed)
    random.seed(seed)


def load_config(config_path: Path) -> dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Dictionary with configuration parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If file is not valid YAML
        
    Example:
        >>> # Assuming config.yaml exists
        >>> config = load_config(Path("config.yaml"))
        >>> config["seed"]
        42
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    return config


def log_experiment(
    experiment_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
) -> None:
    """
    Log experiment parameters and metrics.
    
    Simple logging function that prints to console.
    In real modules, use MLflow for persistent tracking.
    
    Args:
        experiment_name: Name of the experiment
        params: Dictionary of parameters
        metrics: Dictionary of metrics
        
    Example:
        >>> log_experiment(
        ...     "test_exp",
        ...     {"lr": 0.01, "seed": 42},
        ...     {"loss": 0.15, "accuracy": 0.95}
        ... )
        === Experiment: test_exp ===
        Parameters:
          lr: 0.01
          seed: 42
        Metrics:
          loss: 0.15
          accuracy: 0.95
    """
    print(f"=== Experiment: {experiment_name} ===")
    print("Parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


def save_results(
    results: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Save results to YAML file.
    
    Args:
        results: Dictionary to save
        output_path: Output file path
        
    Example:
        >>> results = {"loss": 0.15, "accuracy": 0.95}
        >>> save_results(results, Path("results.yaml"))
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False)
    print(f"Results saved to {output_path}")

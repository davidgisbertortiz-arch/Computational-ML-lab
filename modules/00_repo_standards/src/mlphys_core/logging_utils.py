"""Logging utilities for experiments."""

import logging
import sys
from pathlib import Path
from typing import Any, Optional
import json
from datetime import datetime


def setup_logger(
    name: str = "mlphys",
    level: str = "INFO",
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """
    Setup a logger with console and optional file output.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        
    Returns:
        Configured logger instance
        
    Example:
        >>> logger = setup_logger("my_experiment", level="DEBUG")
        >>> logger.info("Starting experiment...")
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level))
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_metrics(
    metrics: dict[str, Any],
    output_path: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Log metrics to console and optionally to file.
    
    Args:
        metrics: Dictionary of metric names and values
        output_path: Optional path to save metrics as JSON
        logger: Optional logger instance
        
    Example:
        >>> metrics = {"accuracy": 0.95, "loss": 0.15}
        >>> log_metrics(metrics, Path("reports/metrics.json"))
    """
    if logger is None:
        logger = logging.getLogger("mlphys")
    
    logger.info("=== Metrics ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.6f}")
        else:
            logger.info(f"  {key}: {value}")
    
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add timestamp
        metrics_with_timestamp = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
        }
        
        with open(output_path, "w") as f:
            json.dump(metrics_with_timestamp, f, indent=2)
        
        logger.info(f"Metrics saved to {output_path}")


class ExperimentLogger:
    """
    Context manager for experiment logging.
    
    Example:
        >>> with ExperimentLogger("my_exp", level="DEBUG") as logger:
        ...     logger.info("Training started")
        ...     # do work
        ...     logger.info("Training complete")
    """
    
    def __init__(
        self,
        experiment_name: str,
        level: str = "INFO",
        log_file: Optional[Path] = None,
    ):
        self.experiment_name = experiment_name
        self.level = level
        self.log_file = log_file
        self.logger: Optional[logging.Logger] = None
        self.start_time: Optional[datetime] = None
    
    def __enter__(self) -> logging.Logger:
        """Start logging."""
        self.logger = setup_logger(
            name=self.experiment_name,
            level=self.level,
            log_file=self.log_file,
        )
        self.start_time = datetime.now()
        self.logger.info(f"=== Starting experiment: {self.experiment_name} ===")
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End logging."""
        if self.logger and self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            self.logger.info(f"=== Experiment complete in {duration:.2f}s ===")
        
        # Don't suppress exceptions
        return False

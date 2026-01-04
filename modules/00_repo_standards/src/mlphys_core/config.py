"""Configuration management with Pydantic."""

import yaml
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field, field_validator


class ExperimentConfig(BaseModel):
    """Base configuration for all experiments.
    
    All module-specific configs should inherit from this base.
    """
    
    # Experiment metadata
    name: str = Field(..., description="Experiment name")
    description: str = Field(default="", description="Experiment description")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    
    # Output paths
    output_dir: Path = Field(
        default=Path("reports"),
        description="Directory for saving outputs",
    )
    save_artifacts: bool = Field(default=True, description="Whether to save artifacts")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    mlflow_tracking: bool = Field(default=False, description="Enable MLflow tracking")
    
    @field_validator("output_dir", mode="before")
    @classmethod
    def convert_to_path(cls, v: Any) -> Path:
        """Convert string to Path."""
        return Path(v) if not isinstance(v, Path) else v
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v_upper
    
    model_config = {
        "extra": "allow",  # Allow extra fields for module-specific configs
        "arbitrary_types_allowed": True,  # Allow Path and other types
    }


def load_config(config_path: Path, config_class: type[BaseModel] = ExperimentConfig) -> BaseModel:
    """
    Load YAML configuration file and validate with Pydantic.
    
    Args:
        config_path: Path to YAML config file
        config_class: Pydantic model class for validation
        
    Returns:
        Validated configuration object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If file is not valid YAML
        pydantic.ValidationError: If config doesn't match schema
        
    Example:
        >>> config = load_config(Path("configs/experiment.yaml"))
        >>> config.seed
        42
        
        >>> # With custom config class
        >>> class MyConfig(ExperimentConfig):
        ...     learning_rate: float
        >>> config = load_config(Path("configs/custom.yaml"), MyConfig)
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    return config_class(**config_dict)


def save_config(config: BaseModel, output_path: Path) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Pydantic configuration object
        output_path: Output file path
        
    Example:
        >>> config = ExperimentConfig(name="test", seed=42)
        >>> save_config(config, Path("configs/saved.yaml"))
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dict and handle Path objects
    config_dict = config.model_dump()
    
    # Convert Path objects to strings for YAML serialization
    def convert_paths(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: convert_paths(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_paths(item) for item in obj]
        return obj
    
    config_dict = convert_paths(config_dict)
    
    with open(output_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

"""Training configuration using Pydantic."""

from typing import List, Optional, Literal
from pathlib import Path
from pydantic import Field, field_validator
from modules._import_helper import safe_import_from

ExperimentConfig = safe_import_from('00_repo_standards.src.mlphys_core.config', 'ExperimentConfig')


class TrainingConfig(ExperimentConfig):
    """
    Configuration for deep learning training.
    
    Extends base ExperimentConfig with DL-specific parameters.
    All paths are automatically converted to pathlib.Path objects.
    
    Example:
        >>> config = TrainingConfig(
        ...     name="mnist_baseline",
        ...     model_type="CNNMnist",
        ...     batch_size=64,
        ...     num_epochs=10,
        ...     learning_rate=1e-3,
        ... )
        >>> trainer = Trainer(config, model, device)
    """
    
    # ===== Model Architecture =====
    model_type: Literal["SimpleMLP", "CNNMnist"] = Field(
        default="SimpleMLP",
        description="Model architecture to use"
    )
    input_dim: int = Field(
        default=784,
        description="Input dimension (e.g., 784 for MNIST flattened)"
    )
    hidden_dims: List[int] = Field(
        default=[128, 64],
        description="Hidden layer dimensions"
    )
    output_dim: int = Field(
        default=10,
        description="Output dimension (number of classes)"
    )
    dropout: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Dropout probability"
    )
    
    # ===== Training Hyperparameters =====
    batch_size: int = Field(
        default=64,
        gt=0,
        description="Batch size for training"
    )
    num_epochs: int = Field(
        default=10,
        gt=0,
        description="Number of training epochs"
    )
    learning_rate: float = Field(
        default=1e-3,
        gt=0.0,
        description="Learning rate for optimizer"
    )
    weight_decay: float = Field(
        default=0.0,
        ge=0.0,
        description="L2 regularization coefficient"
    )
    optimizer: Literal["adam", "sgd", "adamw"] = Field(
        default="adam",
        description="Optimizer type"
    )
    
    # ===== Engineering Features =====
    use_amp: bool = Field(
        default=False,
        description="Use automatic mixed precision (AMP)"
    )
    gradient_clip_norm: Optional[float] = Field(
        default=None,
        description="Maximum gradient norm (None = no clipping)"
    )
    early_stop_patience: int = Field(
        default=5,
        gt=0,
        description="Patience for early stopping (epochs without improvement)"
    )
    checkpoint_every: int = Field(
        default=1,
        gt=0,
        description="Save checkpoint every N epochs"
    )
    save_best_only: bool = Field(
        default=True,
        description="Only save checkpoint when validation metric improves"
    )
    
    # ===== Data Configuration =====
    dataset: Literal["mnist", "fashion_mnist", "tabular"] = Field(
        default="mnist",
        description="Dataset to use"
    )
    data_dir: Path = Field(
        default=Path("data"),
        description="Directory for dataset downloads"
    )
    val_split: float = Field(
        default=0.1,
        gt=0.0,
        lt=1.0,
        description="Fraction of training data for validation"
    )
    num_workers: int = Field(
        default=4,
        ge=0,
        description="Number of data loading workers"
    )
    
    # ===== Device Configuration =====
    device: Literal["cpu", "cuda", "mps", "auto"] = Field(
        default="auto",
        description="Device to use (auto = use GPU if available)"
    )
    
    # ===== Checkpoint Configuration =====
    checkpoint_dir: Path = Field(
        default=Path("reports/checkpoints"),
        description="Directory for saving checkpoints"
    )
    resume_from: Optional[Path] = Field(
        default=None,
        description="Path to checkpoint to resume from"
    )
    
    @field_validator("hidden_dims")
    @classmethod
    def validate_hidden_dims(cls, v: List[int]) -> List[int]:
        """Ensure all hidden dimensions are positive."""
        if not all(dim > 0 for dim in v):
            raise ValueError("All hidden dimensions must be positive")
        return v
    
    @field_validator("gradient_clip_norm")
    @classmethod
    def validate_gradient_clip(cls, v: Optional[float]) -> Optional[float]:
        """Ensure gradient clip norm is positive if specified."""
        if v is not None and v <= 0:
            raise ValueError("Gradient clip norm must be positive")
        return v

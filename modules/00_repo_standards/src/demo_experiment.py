"""Demo experiment showcasing repository standards."""

import numpy as np
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any
import importlib

# Workaround for Python 3.12+ octal literal parsing
_mlphys_config = importlib.import_module('modules.00_repo_standards.src.mlphys_core.config')
_mlphys_experiment = importlib.import_module('modules.00_repo_standards.src.mlphys_core.experiment')
_mlphys_seeding = importlib.import_module('modules.00_repo_standards.src.mlphys_core.seeding')

ExperimentConfig = _mlphys_config.ExperimentConfig
BaseExperiment = _mlphys_experiment.BaseExperiment
set_seed = _mlphys_seeding.set_seed
from pydantic import Field


class DemoConfig(ExperimentConfig):
    """Configuration for demo experiment."""
    
    # Data generation
    n_samples: int = Field(default=500, description="Number of samples")
    n_features: int = Field(default=20, description="Number of features")
    n_informative: int = Field(default=15, description="Number of informative features")
    n_classes: int = Field(default=2, description="Number of classes")
    test_size: float = Field(default=0.2, description="Test set fraction")
    
    # Model hyperparameters
    C: float = Field(default=1.0, description="Regularization strength")
    max_iter: int = Field(default=1000, description="Maximum iterations")
    solver: str = Field(default="lbfgs", description="Optimization algorithm")


class DemoExperiment(BaseExperiment):
    """
    Demo experiment: Binary classification with sklearn.
    
    Demonstrates:
    - Synthetic data generation
    - Train/test split
    - Model training
    - Metrics logging
    - Plot saving
    - Full reproducibility
    """
    
    def __init__(self, config: DemoConfig):
        # Get module directory
        module_dir = Path(__file__).parent.parent
        super().__init__(config, module_dir)
        self.config: DemoConfig = config
    
    def prepare_data(self) -> dict[str, Any]:
        """Generate synthetic classification data."""
        self.logger.info(
            f"Generating {self.config.n_samples} samples "
            f"with {self.config.n_features} features"
        )
        
        # Generate data (uses global numpy seed)
        X, y = make_classification(
            n_samples=self.config.n_samples,
            n_features=self.config.n_features,
            n_informative=self.config.n_informative,
            n_redundant=self.config.n_features - self.config.n_informative,
            n_classes=self.config.n_classes,
            random_state=self.config.seed,
            shuffle=True,
        )
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.seed,
            stratify=y,
        )
        
        self.logger.info(
            f"Split: {len(X_train)} train, {len(X_test)} test samples"
        )
        
        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }
    
    def build_model(self) -> LogisticRegression:
        """Build logistic regression model."""
        model = LogisticRegression(
            C=self.config.C,
            max_iter=self.config.max_iter,
            solver=self.config.solver,
            random_state=self.config.seed,
        )
        
        self.logger.info(
            f"Created LogisticRegression(C={self.config.C}, "
            f"max_iter={self.config.max_iter}, solver={self.config.solver})"
        )
        
        return model
    
    def train(self, model: LogisticRegression, data: dict) -> LogisticRegression:
        """Train the model."""
        model.fit(data["X_train"], data["y_train"])
        
        self.logger.info(
            f"Training converged in {model.n_iter_[0]} iterations"
        )
        
        return model
    
    def evaluate(self, model: LogisticRegression, data: dict) -> dict[str, Any]:
        """Evaluate model and save visualizations."""
        # Predictions
        y_train_pred = model.predict(data["X_train"])
        y_test_pred = model.predict(data["X_test"])
        
        # Metrics
        train_acc = accuracy_score(data["y_train"], y_train_pred)
        test_acc = accuracy_score(data["y_test"], y_test_pred)
        test_f1 = f1_score(data["y_test"], y_test_pred, average="weighted")
        
        self.logger.info(f"Train accuracy: {train_acc:.4f}")
        self.logger.info(f"Test accuracy: {test_acc:.4f}")
        self.logger.info(f"Test F1 score: {test_f1:.4f}")
        
        # Confusion matrix
        if self.config.save_artifacts:
            self._plot_confusion_matrix(data["y_test"], y_test_pred)
        
        return {
            "train_accuracy": float(train_acc),
            "test_accuracy": float(test_acc),
            "test_f1_score": float(test_f1),
            "n_train": len(data["y_train"]),
            "n_test": len(data["y_test"]),
            "n_iterations": int(model.n_iter_[0]),
        }
    
    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Plot and save confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[f"Class {i}" for i in range(self.config.n_classes)],
            yticklabels=[f"Class {i}" for i in range(self.config.n_classes)],
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        
        # Save
        fig_dir = self.output_dir / "figures"
        fig_dir.mkdir(exist_ok=True)
        fig_path = fig_dir / "confusion_matrix.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        self.logger.info(f"Confusion matrix saved to {fig_path}")

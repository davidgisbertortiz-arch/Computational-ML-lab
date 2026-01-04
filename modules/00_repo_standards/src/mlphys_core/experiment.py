"""Base experiment runner class."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional
import git

# Python 3.12+ workaround - use relative imports within package
from .config import ExperimentConfig, save_config
from .seeding import set_seed
from .logging_utils import setup_logger, log_metrics


class BaseExperiment(ABC):
    """
    Base class for all experiments.
    
    Provides common functionality:
    - Configuration management
    - Seeding
    - Logging
    - Output directory management
    - Git commit tracking
    
    Subclasses should implement:
    - prepare_data()
    - build_model()
    - train()
    - evaluate()
    
    Example:
        >>> class MyExperiment(BaseExperiment):
        ...     def prepare_data(self):
        ...         # Load data
        ...         pass
        ...     def build_model(self):
        ...         # Create model
        ...         pass
        ...     def train(self):
        ...         # Train model
        ...         pass
        ...     def evaluate(self):
        ...         # Evaluate model
        ...         return {"accuracy": 0.95}
        >>> 
        >>> config = ExperimentConfig(name="test", seed=42)
        >>> exp = MyExperiment(config)
        >>> exp.run()
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        module_dir: Optional[Path] = None,
    ):
        """
        Initialize experiment.
        
        Args:
            config: Experiment configuration
            module_dir: Path to module directory (for relative paths)
        """
        self.config = config
        self.module_dir = module_dir or Path.cwd()
        
        # Setup output directory
        if self.config.output_dir.is_absolute():
            self.output_dir = self.config.output_dir
        else:
            self.output_dir = self.module_dir / self.config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        log_file = self.output_dir / f"{self.config.name}.log"
        self.logger = setup_logger(
            name=self.config.name,
            level=self.config.log_level,
            log_file=log_file,
        )
        
        # Set seed
        set_seed(self.config.seed)
        self.logger.info(f"Set random seed: {self.config.seed}")
        
        # Track git commit
        self.git_commit = self._get_git_commit()
        if self.git_commit:
            self.logger.info(f"Git commit: {self.git_commit}")
        
        # Save config
        if self.config.save_artifacts:
            config_path = self.output_dir / "config.yaml"
            save_config(self.config, config_path)
            self.logger.info(f"Config saved to {config_path}")
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            repo = git.Repo(search_parent_directories=True)
            return repo.head.commit.hexsha
        except Exception:
            return None
    
    @abstractmethod
    def prepare_data(self) -> Any:
        """
        Prepare data for the experiment.
        
        Returns:
            Prepared data (format depends on experiment)
        """
        pass
    
    @abstractmethod
    def build_model(self) -> Any:
        """
        Build/initialize the model.
        
        Returns:
            Model instance
        """
        pass
    
    @abstractmethod
    def train(self, model: Any, data: Any) -> Any:
        """
        Train the model.
        
        Args:
            model: Model instance
            data: Training data
            
        Returns:
            Trained model
        """
        pass
    
    @abstractmethod
    def evaluate(self, model: Any, data: Any) -> dict[str, Any]:
        """
        Evaluate the model.
        
        Args:
            model: Trained model
            data: Evaluation data
            
        Returns:
            Dictionary of metrics
        """
        pass
    
    def run(self) -> dict[str, Any]:
        """
        Run the complete experiment pipeline.
        
        Returns:
            Dictionary of final metrics
        """
        self.logger.info(f"=== Running experiment: {self.config.name} ===")
        
        # Pipeline
        self.logger.info("Preparing data...")
        data = self.prepare_data()
        
        self.logger.info("Building model...")
        model = self.build_model()
        
        self.logger.info("Training model...")
        trained_model = self.train(model, data)
        
        self.logger.info("Evaluating model...")
        metrics = self.evaluate(trained_model, data)
        
        # Log and save metrics
        log_metrics(
            metrics,
            output_path=self.output_dir / "metrics.json" if self.config.save_artifacts else None,
            logger=self.logger,
        )
        
        self.logger.info(f"=== Experiment complete ===")
        
        return metrics

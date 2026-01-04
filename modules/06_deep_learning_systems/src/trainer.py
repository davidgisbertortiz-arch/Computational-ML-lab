"""Core training loop implementation."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import time
import json
import logging
from modules._import_helper import safe_import_from

set_seed = safe_import_from('00_repo_standards.src.mlphys_core.seeding', 'set_seed')


class Trainer:
    """
    Training loop with checkpointing, early stopping, and mixed precision.
    
    Features:
    - Config-driven training
    - Automatic checkpointing (best + last)
    - Early stopping based on validation loss
    - Mixed precision training (AMP)
    - Gradient clipping
    - Comprehensive metric logging
    - Resumable training
    
    Args:
        config: TrainingConfig instance
        model: PyTorch model
        device: Device to train on (cpu/cuda/mps)
        logger: Optional logger (creates one if None)
    
    Example:
        >>> from config import TrainingConfig
        >>> from models import SimpleMLP
        >>> 
        >>> config = TrainingConfig(num_epochs=10, learning_rate=1e-3)
        >>> model = SimpleMLP(input_dim=784, hidden_dims=[128, 64], output_dim=10)
        >>> trainer = Trainer(config, model, device="cpu")
        >>> 
        >>> # Train
        >>> history = trainer.train(train_loader, val_loader)
        >>> 
        >>> # Evaluate
        >>> metrics = trainer.evaluate(test_loader)
        >>> print(f"Test accuracy: {metrics['accuracy']:.2%}")
    """
    
    def __init__(
        self,
        config,
        model: nn.Module,
        device: str = "cpu",
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config
        self.model = model.to(device)
        self.device = device
        self.logger = logger or self._create_logger()
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup mixed precision scaler
        self.scaler = GradScaler() if config.use_amp and device == "cuda" else None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
        }
        
        # Ensure checkpoint directory exists
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_logger(self) -> logging.Logger:
        """Create logger for training."""
        logger = logging.getLogger(f"Trainer_{self.config.name}")
        logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer from config."""
        if self.config.optimizer == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with optional AMP
            if self.scaler is not None:
                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.gradient_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                
                # Gradient clipping
                if self.config.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm
                    )
                
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on data loader.
        
        Args:
            data_loader: Data loader for evaluation
        
        Returns:
            Dictionary with 'loss' and 'accuracy' keys
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for data, target in data_loader:
            data, target = data.to(self.device), target.to(self.device)
            
            output = self.model(data)
            loss = self.criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(data_loader)
        accuracy = correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, list]:
        """
        Full training loop with validation and checkpointing.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        
        Returns:
            Training history dictionary
        """
        self.logger.info(f"Starting training for {self.config.num_epochs} epochs")
        self.logger.info(f"Model: {self.config.model_type}, Device: {self.device}")
        self.logger.info(f"Batch size: {self.config.batch_size}, LR: {self.config.learning_rate}")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.evaluate(val_loader)
            val_loss = val_metrics['loss']
            val_accuracy = val_metrics['accuracy']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_accuracy)
            
            # Log progress
            self.logger.info(
                f"Epoch {epoch+1}/{self.config.num_epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Acc: {val_accuracy:.2%}"
            )
            
            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                
                # Save best model
                if self.config.save_best_only:
                    self.save_checkpoint("best_model.pt")
                    self.logger.info(f"✓ New best model saved (val_loss: {val_loss:.4f})")
            else:
                self.epochs_without_improvement += 1
            
            # Regular checkpointing
            if (epoch + 1) % self.config.checkpoint_every == 0:
                if not self.config.save_best_only:
                    self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")
            
            # Early stopping
            if self.epochs_without_improvement >= self.config.early_stop_patience:
                self.logger.info(
                    f"Early stopping triggered after {epoch+1} epochs "
                    f"({self.config.early_stop_patience} epochs without improvement)"
                )
                break
        
        # Save final model
        self.save_checkpoint("last_model.pt")
        
        training_time = time.time() - start_time
        self.logger.info(f"Training complete in {training_time:.1f}s")
        
        return self.history
    
    def save_checkpoint(self, filename: str) -> None:
        """
        Save model checkpoint.
        
        Args:
            filename: Checkpoint filename (will be saved in config.checkpoint_dir)
        """
        checkpoint_path = self.config.checkpoint_dir / filename
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.config.model_dump() if hasattr(self.config, 'model_dump') else dict(self.config),
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.debug(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.history = checkpoint.get('history', {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
        })
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.logger.info(f"Resumed from epoch {self.current_epoch}")
    
    def save_metrics(self, metrics: Dict[str, Any], filename: str = "metrics.json") -> None:
        """
        Save metrics to JSON file.
        
        Args:
            metrics: Dictionary of metrics to save
            filename: Output filename (saved in output_dir)
        """
        output_path = self.config.output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.logger.info(f"Metrics saved to {output_path}")


def train_on_tiny_batch(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    num_steps: int = 200,
    lr: float = 1e-3,
    device: str = "cpu",
) -> Tuple[nn.Module, list]:
    """
    Train model to overfit a tiny batch (for testing).
    
    This is a utility function for engineering tests - it should
    achieve near-zero loss if gradients are flowing correctly.
    
    Args:
        model: PyTorch model
        X: Input features of shape (n_samples, n_features)
        y: Target labels of shape (n_samples,)
        num_steps: Number of optimization steps
        lr: Learning rate
        device: Device to train on
    
    Returns:
        Tuple of (trained_model, loss_history)
    
    Example:
        >>> model = SimpleMLP(input_dim=10, hidden_dims=[20], output_dim=2)
        >>> X, y = torch.randn(8, 10), torch.randint(0, 2, (8,))
        >>> model, losses = train_on_tiny_batch(model, X, y, num_steps=200)
        >>> assert losses[-1] < 0.01  # Should memorize perfectly
    """
    model = model.to(device)
    X, y = X.to(device), y.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    loss_history = []
    model.train()
    
    for step in range(num_steps):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
    
    return model, loss_history

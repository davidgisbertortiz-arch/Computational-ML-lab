"""Dataset loaders for Module 06."""

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from pathlib import Path
from typing import Tuple, Optional
import numpy as np


def get_mnist_loaders(
    data_dir: Path,
    batch_size: int = 64,
    val_split: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create MNIST train/val/test data loaders.
    
    Args:
        data_dir: Directory to download/load MNIST data
        batch_size: Batch size for data loaders
        val_split: Fraction of train set to use for validation
        num_workers: Number of workers for data loading
        seed: Random seed for train/val split
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    
    Example:
        >>> train_loader, val_loader, test_loader = get_mnist_loaders(
        ...     data_dir=Path("data"),
        ...     batch_size=64,
        ...     val_split=0.1,
        ... )
        >>> x, y = next(iter(train_loader))
        >>> assert x.shape == (64, 1, 28, 28)
        >>> assert y.shape == (64,)
    """
    # Ensure data directory exists
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean/std
    ])
    
    # Load datasets
    train_val_dataset = datasets.MNIST(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = datasets.MNIST(
        root=str(data_dir),
        train=False,
        download=True,
        transform=transform
    )
    
    # Split train into train/val
    val_size = int(len(train_val_dataset) * val_split)
    train_size = len(train_val_dataset) - val_size
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        train_val_dataset,
        [train_size, val_size],
        generator=generator
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_fashion_mnist_loaders(
    data_dir: Path,
    batch_size: int = 64,
    val_split: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create FashionMNIST train/val/test data loaders.
    
    Args:
        data_dir: Directory to download/load data
        batch_size: Batch size for data loaders
        val_split: Fraction of train set to use for validation
        num_workers: Number of workers for data loading
        seed: Random seed for train/val split
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Ensure data directory exists
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))  # FashionMNIST mean/std
    ])
    
    # Load datasets
    train_val_dataset = datasets.FashionMNIST(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root=str(data_dir),
        train=False,
        download=True,
        transform=transform
    )
    
    # Split train into train/val
    val_size = int(len(train_val_dataset) * val_split)
    train_size = len(train_val_dataset) - val_size
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        train_val_dataset,
        [train_size, val_size],
        generator=generator
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


class TinyDataset(Dataset):
    """
    Tiny dataset for overfitting tests.
    
    Creates a small synthetic dataset for testing if model can memorize.
    
    Args:
        X: Input features of shape (n_samples, n_features)
        y: Target labels of shape (n_samples,)
    
    Example:
        >>> X = torch.randn(8, 10)
        >>> y = torch.randint(0, 2, (8,))
        >>> dataset = TinyDataset(X, y)
        >>> loader = DataLoader(dataset, batch_size=8)
        >>> x_batch, y_batch = next(iter(loader))
    """
    
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        assert len(X) == len(y), "X and y must have same length"
        self.X = X
        self.y = y
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def create_tiny_dataset(
    n_samples: int = 8,
    n_features: int = 10,
    n_classes: int = 2,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create tiny synthetic dataset for overfitting tests.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_classes: Number of classes
        seed: Random seed
    
    Returns:
        Tuple of (X, y) where X has shape (n_samples, n_features)
        and y has shape (n_samples,)
    
    Example:
        >>> X, y = create_tiny_dataset(n_samples=8, n_features=10, n_classes=2)
        >>> assert X.shape == (8, 10)
        >>> assert y.shape == (8,)
        >>> assert y.min() >= 0 and y.max() < 2
    """
    torch.manual_seed(seed)
    X = torch.randn(n_samples, n_features)
    y = torch.randint(0, n_classes, (n_samples,))
    return X, y


def get_data_loaders(config):
    """
    Factory function to create data loaders from config.
    
    Args:
        config: TrainingConfig instance
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    
    Example:
        >>> from config import TrainingConfig
        >>> config = TrainingConfig(dataset="mnist", batch_size=64)
        >>> train_loader, val_loader, test_loader = get_data_loaders(config)
    """
    if config.dataset == "mnist":
        return get_mnist_loaders(
            data_dir=config.data_dir,
            batch_size=config.batch_size,
            val_split=config.val_split,
            num_workers=config.num_workers,
            seed=config.seed,
        )
    elif config.dataset == "fashion_mnist":
        return get_fashion_mnist_loaders(
            data_dir=config.data_dir,
            batch_size=config.batch_size,
            val_split=config.val_split,
            num_workers=config.num_workers,
            seed=config.seed,
        )
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

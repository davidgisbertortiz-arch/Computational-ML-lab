"""Pytest fixtures for Module 06 tests."""

import pytest
import torch
import numpy as np
from pathlib import Path
from modules._import_helper import safe_import_from

set_seed = safe_import_from('00_repo_standards.src.mlphys_core.seeding', 'set_seed')


@pytest.fixture(autouse=True)
def reset_seeds():
    """Reset all random seeds before each test."""
    set_seed(42)
    yield
    

@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Create temporary directory for checkpoints."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    return checkpoint_dir


@pytest.fixture
def tiny_dataset():
    """Create tiny dataset for overfitting tests."""
    torch.manual_seed(42)
    X = torch.randn(8, 10)
    y = torch.randint(0, 2, (8,))
    return X, y


@pytest.fixture
def simple_model():
    """Create simple MLP for testing."""
    torch.manual_seed(42)
    return torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 2)
    )

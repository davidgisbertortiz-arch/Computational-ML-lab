"""Seeding utilities for reproducibility."""

import random
import numpy as np
from typing import Optional


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across all libraries.
    
    Sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (if available)
    
    Args:
        seed: Random seed (integer)
        
    Example:
        >>> set_seed(42)
        >>> np.random.randn(1)
        array([0.49671415])
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Set PyTorch seed if available
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # Make PyTorch deterministic (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_rng(seed: Optional[int] = None) -> np.random.Generator:
    """
    Get a NumPy random number generator.
    
    Preferred over np.random.seed() for better isolation and reproducibility.
    
    Args:
        seed: Random seed. If None, uses a random seed.
        
    Returns:
        NumPy random Generator instance
        
    Example:
        >>> rng = get_rng(42)
        >>> rng.standard_normal(5)
        array([ 0.30471708, -1.03998411,  0.7504512 ,  0.94056472, -1.95103519])
    """
    return np.random.default_rng(seed)


def check_determinism(fn: callable, seed: int, n_runs: int = 3) -> bool:
    """
    Check if a function produces deterministic results with the same seed.
    
    Useful for testing reproducibility of experiments.
    
    Args:
        fn: Function to test (should accept no arguments)
        seed: Random seed to use
        n_runs: Number of runs to test
        
    Returns:
        True if all runs produce identical results, False otherwise
        
    Example:
        >>> def my_experiment():
        ...     return np.random.randn(10).tolist()
        >>> check_determinism(my_experiment, seed=42, n_runs=3)
        True
    """
    results = []
    
    for _ in range(n_runs):
        set_seed(seed)
        result = fn()
        results.append(result)
    
    # Check all results match the first one
    first_result = results[0]
    
    for result in results[1:]:
        # Handle different types
        if isinstance(first_result, np.ndarray):
            if not np.array_equal(result, first_result):
                return False
        elif isinstance(first_result, (list, tuple)):
            if result != first_result:
                return False
        elif isinstance(first_result, dict):
            if result != first_result:
                return False
        else:
            if result != first_result:
                return False
    
    return True

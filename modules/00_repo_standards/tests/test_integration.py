"""Integration tests for Module 00."""

import pytest
import numpy as np
from pathlib import Path
from modules._import_helper import safe_import_from

gradient_descent = safe_import_from('00_repo_standards.src.core', 'gradient_descent')
set_seed, load_config = safe_import_from('00_repo_standards.src.utils', 'set_seed', 'load_config')


class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_full_experiment_pipeline(self):
        """Test complete experiment workflow."""
        # 1. Set seed
        set_seed(42)
        
        # 2. Setup problem
        dim = 5
        x0 = np.random.randn(dim)
        
        # 3. Run optimization
        tol = 1e-6
        x_opt, history = gradient_descent(
            x0=x0,
            grad_fn=lambda x: 2 * x,  # gradient of f(x) = ||x||^2
            lr=0.1,
            max_iter=100,
            tol=tol,
        )
        
        # 5. Check convergence
        assert np.linalg.norm(x_opt) < 1e-3
        assert history[-1] < tol
    
    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results."""
        dim = 3
        lr = 0.1
        
        # Run 1
        set_seed(42)
        x0_1 = np.random.randn(dim)
        x_opt_1, _ = gradient_descent(
            x0=x0_1,
            grad_fn=lambda x: 2 * x,
            lr=lr,
            max_iter=100,
        )
        
        # Run 2 with same seed
        set_seed(42)
        x0_2 = np.random.randn(dim)
        x_opt_2, _ = gradient_descent(
            x0=x0_2,
            grad_fn=lambda x: 2 * x,
            lr=lr,
            max_iter=100,
        )
        
        # Should produce identical results
        assert np.array_equal(x0_1, x0_2)
        assert np.array_equal(x_opt_1, x_opt_2)


class TestSanityChecks:
    """Sanity tests that optimization actually works."""
    
    def test_optimizer_decreases_loss(self):
        """Test that optimizer reduces objective value."""
        def objective(x):
            return np.sum(x**2)
        
        def grad_fn(x):
            return 2 * x
        
        x0 = np.random.randn(10) * 10  # Start far from optimum
        
        x_opt, history = gradient_descent(
            x0=x0,
            grad_fn=grad_fn,
            lr=0.1,
            max_iter=200,
        )
        
        initial_loss = objective(x0)
        final_loss = objective(x_opt)
        
        # Final loss should be much smaller
        assert final_loss < initial_loss * 0.01
    
    def test_can_find_known_optimum(self):
        """Test that optimizer finds known minimum."""
        # Minimize f(x) = (x - 3)^2, minimum at x=3
        def grad_fn(x):
            return 2 * (x - 3)
        
        x0 = np.array([0.0])
        x_opt, _ = gradient_descent(
            x0=x0,
            grad_fn=grad_fn,
            lr=0.1,
            max_iter=200,
        )
        
        # Should find x ≈ 3
        assert np.allclose(x_opt, 3.0, atol=1e-2)

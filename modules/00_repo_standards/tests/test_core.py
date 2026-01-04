"""Tests for core mathematical operations."""

import pytest
import numpy as np
from modules._import_helper import safe_import_from

(compute_mean, compute_variance, gradient_descent, numerical_gradient) = safe_import_from(
    '00_repo_standards.src.core',
    'compute_mean', 'compute_variance', 'gradient_descent', 'numerical_gradient'
)


class TestComputeMean:
    """Tests for compute_mean function."""
    
    def test_simple_mean(self):
        """Test mean of simple array."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert compute_mean(x) == 3.0
    
    def test_single_element(self):
        """Test mean of single element."""
        x = np.array([42.0])
        assert compute_mean(x) == 42.0
    
    def test_negative_values(self):
        """Test mean with negative values."""
        x = np.array([-1.0, 0.0, 1.0])
        assert compute_mean(x) == 0.0
    
    def test_empty_array_raises(self):
        """Test that empty array raises ValueError."""
        with pytest.raises(ValueError, match="empty array"):
            compute_mean(np.array([]))


class TestComputeVariance:
    """Tests for compute_variance function."""
    
    def test_simple_variance(self):
        """Test variance of simple array."""
        x = np.array([1.0, 2.0, 3.0])
        # Population variance: ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = 2/3
        assert np.isclose(compute_variance(x), 2/3)
    
    def test_sample_variance(self):
        """Test sample variance (ddof=1)."""
        x = np.array([1.0, 2.0, 3.0])
        # Sample variance: 2/2 = 1.0
        assert np.isclose(compute_variance(x, ddof=1), 1.0)
    
    def test_constant_array(self):
        """Test variance of constant array."""
        x = np.array([5.0, 5.0, 5.0])
        assert compute_variance(x) == 0.0
    
    def test_empty_array_raises(self):
        """Test that empty array raises ValueError."""
        with pytest.raises(ValueError, match="empty array"):
            compute_variance(np.array([]))
    
    def test_insufficient_samples_for_ddof_raises(self):
        """Test that ddof >= size raises ValueError."""
        x = np.array([1.0])
        with pytest.raises(ValueError, match="insufficient"):
            compute_variance(x, ddof=1)


class TestGradientDescent:
    """Tests for gradient_descent optimizer."""
    
    def test_converges_on_quadratic(self):
        """Test that GD converges on simple quadratic."""
        # Minimize f(x) = x^2, gradient = 2x
        def grad_fn(x):
            return 2 * x
        
        x0 = np.array([1.0])
        x_opt, history = gradient_descent(
            x0=x0,
            grad_fn=grad_fn,
            lr=0.1,
            max_iter=100,
            tol=1e-6,
        )
        
        # Should converge to 0
        assert np.allclose(x_opt, 0.0, atol=1e-3)
        # Should converge (grad norm below tol)
        assert history[-1] < 1e-6
    
    def test_multidimensional(self):
        """Test GD on multidimensional problem."""
        # Minimize f(x) = ||x||^2, gradient = 2x
        def grad_fn(x):
            return 2 * x
        
        x0 = np.random.randn(5)
        x_opt, history = gradient_descent(
            x0=x0,
            grad_fn=grad_fn,
            lr=0.1,
            max_iter=500,
            tol=1e-6,
        )
        
        assert np.allclose(x_opt, 0.0, atol=1e-3)
    
    def test_records_history(self):
        """Test that history is recorded correctly."""
        def grad_fn(x):
            return 2 * x
        
        x0 = np.array([1.0])
        x_opt, history = gradient_descent(
            x0=x0,
            grad_fn=grad_fn,
            lr=0.1,
            max_iter=10,
            tol=1e-10,  # Won't converge in 10 iters
        )
        
        # Should have 10 entries in history
        assert len(history) == 10
        # Gradient norms should decrease
        assert history[-1] < history[0]
    
    def test_invalid_lr_raises(self):
        """Test that negative lr raises ValueError."""
        with pytest.raises(ValueError, match="Learning rate"):
            gradient_descent(
                x0=np.array([1.0]),
                grad_fn=lambda x: 2*x,
                lr=-0.1,
            )
    
    def test_invalid_max_iter_raises(self):
        """Test that non-positive max_iter raises ValueError."""
        with pytest.raises(ValueError, match="max_iter"):
            gradient_descent(
                x0=np.array([1.0]),
                grad_fn=lambda x: 2*x,
                max_iter=0,
            )


class TestNumericalGradient:
    """Tests for numerical_gradient function."""
    
    def test_quadratic_gradient(self):
        """Test numerical gradient on quadratic function."""
        # f(x) = ||x||^2, analytical gradient = 2x
        fn = lambda x: np.sum(x**2)
        x = np.array([1.0, 2.0, 3.0])
        
        num_grad = numerical_gradient(fn, x)
        analytical_grad = 2 * x
        
        assert np.allclose(num_grad, analytical_grad, rtol=1e-4)
    
    def test_linear_gradient(self):
        """Test numerical gradient on linear function."""
        # f(x) = a^T x, gradient = a
        a = np.array([1.0, 2.0, 3.0])
        fn = lambda x: np.dot(a, x)
        x = np.array([1.0, 1.0, 1.0])
        
        num_grad = numerical_gradient(fn, x)
        
        assert np.allclose(num_grad, a, rtol=1e-4)
    
    def test_single_dimension(self):
        """Test numerical gradient in 1D."""
        # f(x) = x^3, gradient = 3x^2
        fn = lambda x: x[0]**3
        x = np.array([2.0])
        
        num_grad = numerical_gradient(fn, x)
        analytical_grad = 3 * x[0]**2
        
        assert np.allclose(num_grad, analytical_grad, rtol=1e-4)


@pytest.mark.parametrize("dim", [1, 2, 5, 10])
def test_gradient_descent_dimensions(dim):
    """Test gradient descent works for various dimensions."""
    def grad_fn(x):
        return 2 * x
    
    x0 = np.random.randn(dim)
    x_opt, _ = gradient_descent(
        x0=x0,
        grad_fn=grad_fn,
        lr=0.1,
        max_iter=200,
        tol=1e-6,
    )
    
    assert x_opt.shape == (dim,)
    assert np.allclose(x_opt, 0.0, atol=1e-3)

"""Core mathematical operations and algorithms."""

import numpy as np
from typing import Callable


def compute_mean(x: np.ndarray) -> float:
    """
    Compute arithmetic mean of array.
    
    Args:
        x: Input array of shape (n,)
        
    Returns:
        Mean value as float
        
    Raises:
        ValueError: If array is empty
        
    Example:
        >>> compute_mean(np.array([1, 2, 3]))
        2.0
    """
    if x.size == 0:
        raise ValueError("Cannot compute mean of empty array")
    return float(np.mean(x))


def compute_variance(x: np.ndarray, ddof: int = 0) -> float:
    """
    Compute variance of array.
    
    Args:
        x: Input array of shape (n,)
        ddof: Delta degrees of freedom (0=population, 1=sample)
        
    Returns:
        Variance as float
        
    Raises:
        ValueError: If array is empty or has insufficient samples
        
    Example:
        >>> compute_variance(np.array([1, 2, 3]))
        0.6666666666666666
    """
    if x.size == 0:
        raise ValueError("Cannot compute variance of empty array")
    if ddof > 0 and x.size <= ddof:
        raise ValueError(f"Array size {x.size} insufficient for ddof={ddof}")
    return float(np.var(x, ddof=ddof))


def gradient_descent(
    x0: np.ndarray,
    grad_fn: Callable[[np.ndarray], np.ndarray],
    lr: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-6,
    verbose: bool = False,
) -> tuple[np.ndarray, list[float]]:
    """
    Gradient descent optimizer.
    
    Minimizes a function by following the negative gradient direction.
    Stops when gradient norm is below tolerance or max iterations reached.
    
    Args:
        x0: Initial point (shape: [n_features])
        grad_fn: Function computing gradient at point x
        lr: Learning rate (step size), must be positive
        max_iter: Maximum number of iterations
        tol: Convergence tolerance on gradient norm
        verbose: If True, print progress every 100 iterations
        
    Returns:
        Tuple of (optimal_point, loss_history)
        - optimal_point: Final point after optimization
        - loss_history: List of gradient norms at each iteration
        
    Raises:
        ValueError: If lr <= 0 or max_iter <= 0
        
    Example:
        >>> # Minimize f(x) = x^2, gradient = 2x
        >>> def grad_fn(x): return 2 * x
        >>> x_opt, history = gradient_descent(
        ...     x0=np.array([1.0]),
        ...     grad_fn=grad_fn,
        ...     lr=0.1,
        ...     max_iter=100,
        ... )
        >>> abs(x_opt[0]) < 1e-3  # Should converge to 0
        True
    """
    if lr <= 0:
        raise ValueError(f"Learning rate must be positive, got {lr}")
    if max_iter <= 0:
        raise ValueError(f"max_iter must be positive, got {max_iter}")
    
    x = x0.copy()
    history = []
    
    for i in range(max_iter):
        grad = grad_fn(x)
        grad_norm = float(np.linalg.norm(grad))
        history.append(grad_norm)
        
        if verbose and i % 100 == 0:
            print(f"Iteration {i}: grad_norm = {grad_norm:.6f}")
        
        if grad_norm < tol:
            if verbose:
                print(f"Converged at iteration {i}")
            break
        
        # Update step
        x = x - lr * grad
    
    return x, history


def numerical_gradient(
    fn: Callable[[np.ndarray], float],
    x: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Compute numerical gradient using finite differences.
    
    Uses central differences: grad[i] ≈ (f(x + eps*e_i) - f(x - eps*e_i)) / (2*eps)
    
    Args:
        fn: Scalar function to differentiate
        x: Point at which to compute gradient
        eps: Finite difference step size
        
    Returns:
        Numerical gradient at x
        
    Example:
        >>> fn = lambda x: np.sum(x**2)  # f(x) = ||x||^2
        >>> x = np.array([1.0, 2.0])
        >>> grad = numerical_gradient(fn, x)
        >>> np.allclose(grad, 2*x)  # Gradient should be 2x
        True
    """
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        grad[i] = (fn(x_plus) - fn(x_minus)) / (2 * eps)
    return grad

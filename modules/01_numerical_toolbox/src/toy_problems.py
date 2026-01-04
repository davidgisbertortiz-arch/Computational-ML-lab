"""Toy problems for testing and demonstrating numerical optimization.

This module provides simple, well-understood problems for:
- Testing optimizer implementations
- Demonstrating conditioning effects
- Comparing closed-form vs iterative solutions
"""

import numpy as np
from typing import Callable, Tuple, Optional
from dataclasses import dataclass


@dataclass
class QuadraticBowl:
    """
    Quadratic function f(x) = 0.5 * x^T A x - b^T x + c
    
    Attributes:
        A: Positive definite matrix (Hessian)
        b: Linear coefficient vector
        c: Constant offset
        condition_number: kappa(A) = lambda_max / lambda_min
        optimum: x* = A^{-1} b (global minimum)
        optimum_value: f(x*)
    """
    A: np.ndarray
    b: np.ndarray
    c: float
    condition_number: float
    optimum: np.ndarray
    optimum_value: float
    
    def __call__(self, x: np.ndarray) -> float:
        """Evaluate f(x)."""
        return 0.5 * x @ self.A @ x - self.b @ x + self.c
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute gradient: ∇f(x) = Ax - b."""
        return self.A @ x - self.b
    
    def hessian(self, x: np.ndarray) -> np.ndarray:
        """Hessian is constant: H(x) = A."""
        return self.A


def create_quadratic_bowl(
    n_dim: int = 2,
    condition_number: float = 10.0,
    seed: int = 42,
    offset: Optional[np.ndarray] = None,
) -> QuadraticBowl:
    """
    Create a quadratic bowl with specified condition number.
    
    Args:
        n_dim: Problem dimensionality
        condition_number: Target condition number (lambda_max / lambda_min)
        seed: Random seed for reproducibility
        offset: Location of minimum (default: random)
        
    Returns:
        QuadraticBowl instance
        
    Construction:
        1. Generate eigenvalues with geometric spacing
        2. Create random orthogonal matrix Q
        3. A = Q * diag(eigenvalues) * Q^T
        4. Choose random b (determines minimum location)
        
    Example:
        >>> bowl = create_quadratic_bowl(n_dim=2, condition_number=100)
        >>> print(f"Condition: {bowl.condition_number:.1f}")
        >>> x = np.array([1.0, 0.0])
        >>> f_val = bowl(x)
        >>> grad = bowl.gradient(x)
    """
    np.random.seed(seed)
    
    # Create eigenvalues with geometric spacing
    eigenvalues = np.logspace(0, -np.log10(condition_number), n_dim)
    
    # Shuffle to avoid ordering bias
    np.random.shuffle(eigenvalues)
    
    # Create random rotation
    Q, _ = np.linalg.qr(np.random.randn(n_dim, n_dim))
    
    # Construct A = Q Lambda Q^T
    A = Q @ np.diag(eigenvalues) @ Q.T
    
    # Ensure symmetry (numerical stability)
    A = 0.5 * (A + A.T)
    
    # Create b (determines minimum location)
    if offset is None:
        offset = np.random.randn(n_dim)
    
    b = A @ offset  # So that A^{-1} b = offset
    
    # Constant term
    c = 0.0
    
    # Compute optimum
    optimum = np.linalg.solve(A, b)
    optimum_value = 0.5 * optimum @ A @ optimum - b @ optimum + c
    
    # Verify condition number
    actual_kappa = np.linalg.cond(A)
    
    return QuadraticBowl(
        A=A,
        b=b,
        c=c,
        condition_number=actual_kappa,
        optimum=optimum,
        optimum_value=optimum_value,
    )


@dataclass
class LinearRegressionProblem:
    """
    Linear regression: minimize ||Xw - y||^2
    
    Attributes:
        X: Design matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        true_weights: Ground truth (if synthetic)
        condition_number: kappa(X^T X)
    """
    X: np.ndarray
    y: np.ndarray
    true_weights: Optional[np.ndarray]
    condition_number: float
    
    def objective(self, w: np.ndarray) -> float:
        """MSE loss: (1/n) ||Xw - y||^2."""
        residuals = self.X @ w - self.y
        return np.mean(residuals ** 2)
    
    def gradient(self, w: np.ndarray) -> np.ndarray:
        """Gradient: (2/n) X^T (Xw - y)."""
        residuals = self.X @ w - self.y
        return (2.0 / len(self.y)) * self.X.T @ residuals
    
    def hessian(self, w: np.ndarray) -> np.ndarray:
        """Hessian: (2/n) X^T X."""
        return (2.0 / len(self.y)) * self.X.T @ self.X


def create_linear_regression(
    n_samples: int = 100,
    n_features: int = 10,
    noise_std: float = 0.1,
    condition_number: Optional[float] = None,
    seed: int = 42,
) -> LinearRegressionProblem:
    """
    Create a synthetic linear regression problem.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        noise_std: Standard deviation of Gaussian noise
        condition_number: If specified, create ill-conditioned X
        seed: Random seed
        
    Returns:
        LinearRegressionProblem instance
        
    Construction:
        - If condition_number specified: X = U Sigma V^T with controlled spectrum
        - Otherwise: X ~ N(0, 1)
        - y = X @ w_true + noise
    
    Example:
        >>> prob = create_linear_regression(
        ...     n_samples=200, n_features=5, condition_number=100
        ... )
        >>> w_closed = linear_regression_closed_form(prob.X, prob.y)
        >>> loss = prob.objective(w_closed)
    """
    np.random.seed(seed)
    
    # Create design matrix
    if condition_number is not None:
        # Create ill-conditioned matrix
        # Note: κ(X^T X) = κ(X)^2, so we need κ(X) = sqrt(condition_number)
        U, _ = np.linalg.qr(np.random.randn(n_samples, n_features))
        V, _ = np.linalg.qr(np.random.randn(n_features, n_features))
        
        # Use sqrt so that X^T X has the target condition number
        kappa_X = np.sqrt(condition_number)
        singular_values = np.logspace(0, -np.log10(kappa_X), n_features)
        Sigma = np.diag(singular_values)
        
        X = U @ Sigma @ V.T
    else:
        X = np.random.randn(n_samples, n_features)
    
    # Create true weights
    true_weights = np.random.randn(n_features)
    
    # Generate targets with noise
    y = X @ true_weights + noise_std * np.random.randn(n_samples)
    
    # Compute condition number
    kappa = np.linalg.cond(X.T @ X)
    
    return LinearRegressionProblem(
        X=X,
        y=y,
        true_weights=true_weights,
        condition_number=kappa,
    )


def linear_regression_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Solve linear regression via normal equations: w = (X^T X)^{-1} X^T y
    
    Args:
        X: Design matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        
    Returns:
        Optimal weights (n_features,)
        
    Note:
        - Exact solution for well-conditioned problems
        - Can be numerically unstable if X^T X is ill-conditioned
        - O(n_features^3) complexity
    
    Example:
        >>> X = np.random.randn(100, 5)
        >>> y = np.random.randn(100)
        >>> w = linear_regression_closed_form(X, y)
        >>> print(w.shape)  # (5,)
    """
    # More stable: use np.linalg.lstsq which uses SVD
    w, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    return w


def linear_regression_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, list]:
    """
    Solve linear regression via gradient descent.
    
    Args:
        X: Design matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        learning_rate: Step size (alpha)
        max_iter: Maximum iterations
        tol: Convergence tolerance on gradient norm
        
    Returns:
        Tuple of (optimal_weights, loss_history)
        
    Algorithm:
        w_{t+1} = w_t - alpha * gradient(w_t)
        where gradient(w) = (2/n) X^T (Xw - y)
    
    Example:
        >>> X = np.random.randn(100, 5)
        >>> y = np.random.randn(100)
        >>> w, losses = linear_regression_gradient_descent(X, y, learning_rate=0.1)
        >>> print(f"Final loss: {losses[-1]:.6f}")
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    loss_history = []
    
    for iteration in range(max_iter):
        # Compute gradient
        residuals = X @ w - y
        grad = (2.0 / n_samples) * X.T @ residuals
        
        # Update weights
        w = w - learning_rate * grad
        
        # Track loss
        loss = np.mean(residuals ** 2)
        loss_history.append(loss)
        
        # Check convergence
        if np.linalg.norm(grad) < tol:
            break
    
    return w, loss_history


def rosenbrock_function(x: np.ndarray, a: float = 1.0, b: float = 100.0) -> float:
    """
    Rosenbrock function: f(x,y) = (a-x)^2 + b(y-x^2)^2
    
    Classic non-convex optimization test problem.
    Global minimum at (a, a^2) with value 0.
    
    Args:
        x: Point (2D array)
        a: Parameter (default 1.0)
        b: Parameter (default 100.0)
        
    Returns:
        Function value
        
    Properties:
        - Non-convex
        - Global minimum at (1, 1)
        - Narrow curved valley
        - Tests optimizer robustness
    
    Example:
        >>> x = np.array([0.0, 0.0])
        >>> f_val = rosenbrock_function(x)
        >>> print(f_val)  # Should be 1.0
    """
    if len(x) != 2:
        raise ValueError("Rosenbrock function requires 2D input")
    
    x_val, y_val = x[0], x[1]
    return (a - x_val) ** 2 + b * (y_val - x_val ** 2) ** 2


def rosenbrock_gradient(x: np.ndarray, a: float = 1.0, b: float = 100.0) -> np.ndarray:
    """
    Gradient of Rosenbrock function.
    
    ∇f = [-2(a-x) - 4bx(y-x^2), 2b(y-x^2)]
    
    Args:
        x: Point (2D array)
        a: Parameter
        b: Parameter
        
    Returns:
        Gradient vector (2D)
    """
    if len(x) != 2:
        raise ValueError("Rosenbrock function requires 2D input")
    
    x_val, y_val = x[0], x[1]
    
    grad_x = -2.0 * (a - x_val) - 4.0 * b * x_val * (y_val - x_val ** 2)
    grad_y = 2.0 * b * (y_val - x_val ** 2)
    
    return np.array([grad_x, grad_y])


def beale_function(x: np.ndarray) -> float:
    """
    Beale function: f(x,y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2
    
    Global minimum at (3, 0.5) with value 0.
    
    Args:
        x: Point (2D array)
        
    Returns:
        Function value
        
    Properties:
        - Non-convex
        - Multiple local minima
        - Tests optimizer robustness
    """
    if len(x) != 2:
        raise ValueError("Beale function requires 2D input")
    
    x_val, y_val = x[0], x[1]
    
    term1 = (1.5 - x_val + x_val * y_val) ** 2
    term2 = (2.25 - x_val + x_val * y_val ** 2) ** 2
    term3 = (2.625 - x_val + x_val * y_val ** 3) ** 2
    
    return term1 + term2 + term3


def beale_gradient(x: np.ndarray) -> np.ndarray:
    """
    Gradient of Beale function.
    
    Args:
        x: Point (2D array)
        
    Returns:
        Gradient vector (2D)
    """
    if len(x) != 2:
        raise ValueError("Beale function requires 2D input")
    
    x_val, y_val = x[0], x[1]
    
    # Compute terms
    t1 = 1.5 - x_val + x_val * y_val
    t2 = 2.25 - x_val + x_val * y_val ** 2
    t3 = 2.625 - x_val + x_val * y_val ** 3
    
    # Partial derivatives
    grad_x = (
        2 * t1 * (-1 + y_val)
        + 2 * t2 * (-1 + y_val ** 2)
        + 2 * t3 * (-1 + y_val ** 3)
    )
    
    grad_y = (
        2 * t1 * x_val
        + 2 * t2 * x_val * 2 * y_val
        + 2 * t3 * x_val * 3 * y_val ** 2
    )
    
    return np.array([grad_x, grad_y])

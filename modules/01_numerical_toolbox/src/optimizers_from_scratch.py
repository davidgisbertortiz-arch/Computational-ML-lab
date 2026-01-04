"""Gradient-based optimizers implemented from scratch.

This module provides clean implementations of fundamental optimization algorithms:
- Gradient Descent (GD)
- Momentum
- Adam (Adaptive Moment Estimation)

Each optimizer tracks convergence metrics for analysis.
"""

import numpy as np
from typing import Callable, Optional
from dataclasses import dataclass, field
import typer


@dataclass
class OptimizationResult:
    """Container for optimization results and diagnostics."""
    
    x_final: np.ndarray
    f_final: float
    history: dict[str, list]
    converged: bool
    n_iterations: int
    
    def __repr__(self) -> str:
        status = "converged" if self.converged else "did not converge"
        return (
            f"OptimizationResult({status} in {self.n_iterations} iterations, "
            f"f={self.f_final:.6e})"
        )


class GradientDescent:
    """
    Vanilla Gradient Descent optimizer.
    
    Update rule: x_{t+1} = x_t - alpha * grad_f(x_t)
    
    Args:
        learning_rate: Step size (default: 0.01)
        max_iter: Maximum iterations (default: 1000)
        tol: Convergence tolerance on gradient norm (default: 1e-6)
        verbose: Print progress every verbose iterations (default: 0 = no print)
    
    Example:
        >>> def quadratic(x): return 0.5 * np.sum(x**2)
        >>> def grad_quadratic(x): return x
        >>> opt = GradientDescent(learning_rate=0.1)
        >>> result = opt.minimize(quadratic, grad_quadratic, x0=np.array([1.0, 2.0]))
        >>> print(result)
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-6,
        verbose: int = 0,
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
    
    def minimize(
        self,
        f: Callable[[np.ndarray], float],
        grad_f: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
    ) -> OptimizationResult:
        """
        Minimize function f starting from x0.
        
        Args:
            f: Objective function
            grad_f: Gradient of objective
            x0: Initial point
            
        Returns:
            OptimizationResult with final point and diagnostics
        """
        x = x0.copy()
        history = {
            "f_vals": [],
            "grad_norms": [],
            "step_sizes": [],
        }
        
        for iteration in range(self.max_iter):
            # Evaluate
            f_val = f(x)
            grad = grad_f(x)
            grad_norm = np.linalg.norm(grad)
            
            # Log
            history["f_vals"].append(float(f_val))
            history["grad_norms"].append(float(grad_norm))
            history["step_sizes"].append(float(self.learning_rate * grad_norm))
            
            # Verbose output
            if self.verbose > 0 and iteration % self.verbose == 0:
                print(f"  Iter {iteration:4d}: f={f_val:.6e}, |grad|={grad_norm:.6e}")
            
            # Check convergence
            if grad_norm < self.tol:
                if self.verbose > 0:
                    print(f"  Converged at iteration {iteration}")
                return OptimizationResult(
                    x_final=x,
                    f_final=float(f_val),
                    history=history,
                    converged=True,
                    n_iterations=iteration + 1,
                )
            
            # Update step
            x = x - self.learning_rate * grad
        
        # Max iterations reached
        f_final = f(x)
        history["f_vals"].append(float(f_final))
        history["grad_norms"].append(float(np.linalg.norm(grad_f(x))))
        
        return OptimizationResult(
            x_final=x,
            f_final=float(f_final),
            history=history,
            converged=False,
            n_iterations=self.max_iter,
        )


class MomentumOptimizer:
    """
    Gradient Descent with Momentum.
    
    Update rules:
        v_{t+1} = beta * v_t + grad_f(x_t)
        x_{t+1} = x_t - alpha * v_{t+1}
    
    Args:
        learning_rate: Step size (default: 0.01)
        momentum: Momentum coefficient beta (default: 0.9)
        max_iter: Maximum iterations (default: 1000)
        tol: Convergence tolerance (default: 1e-6)
        verbose: Print frequency (default: 0)
    
    Example:
        >>> opt = MomentumOptimizer(learning_rate=0.01, momentum=0.9)
        >>> result = opt.minimize(f, grad_f, x0)
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        max_iter: int = 1000,
        tol: float = 1e-6,
        verbose: int = 0,
    ):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
    
    def minimize(
        self,
        f: Callable[[np.ndarray], float],
        grad_f: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
    ) -> OptimizationResult:
        """Minimize function with momentum."""
        x = x0.copy()
        v = np.zeros_like(x)  # velocity
        
        history = {
            "f_vals": [],
            "grad_norms": [],
            "velocity_norms": [0.0],  # Initial velocity norm (v starts at zero)
        }
        
        for iteration in range(self.max_iter):
            f_val = f(x)
            grad = grad_f(x)
            grad_norm = np.linalg.norm(grad)
            
            # Log
            history["f_vals"].append(float(f_val))
            history["grad_norms"].append(float(grad_norm))
            
            if self.verbose > 0 and iteration % self.verbose == 0:
                print(
                    f"  Iter {iteration:4d}: f={f_val:.6e}, "
                    f"|grad|={grad_norm:.6e}, |v|={np.linalg.norm(v):.6e}"
                )
            
            # Check convergence
            if grad_norm < self.tol:
                if self.verbose > 0:
                    print(f"  Converged at iteration {iteration}")
                return OptimizationResult(
                    x_final=x,
                    f_final=float(f_val),
                    history=history,
                    converged=True,
                    n_iterations=iteration + 1,
                )
            
            # Momentum update
            v = self.momentum * v + grad
            x = x - self.learning_rate * v
            
            # Log velocity after update
            history["velocity_norms"].append(float(np.linalg.norm(v)))
        
        # Max iterations
        f_final = f(x)
        history["f_vals"].append(float(f_final))
        
        return OptimizationResult(
            x_final=x,
            f_final=float(f_final),
            history=history,
            converged=False,
            n_iterations=self.max_iter,
        )


class AdamOptimizer:
    """
    Adam (Adaptive Moment Estimation) optimizer.
    
    Update rules:
        m_t = beta1 * m_{t-1} + (1-beta1) * grad
        v_t = beta2 * v_{t-1} + (1-beta2) * grad^2
        m_hat = m_t / (1 - beta1^t)
        v_hat = v_t / (1 - beta2^t)
        x_{t+1} = x_t - alpha * m_hat / (sqrt(v_hat) + epsilon)
    
    Args:
        learning_rate: Step size (default: 0.001)
        beta1: First moment decay (default: 0.9)
        beta2: Second moment decay (default: 0.999)
        epsilon: Numerical stability constant (default: 1e-8)
        max_iter: Maximum iterations (default: 1000)
        tol: Convergence tolerance (default: 1e-6)
        verbose: Print frequency (default: 0)
    
    Example:
        >>> opt = AdamOptimizer(learning_rate=0.001)
        >>> result = opt.minimize(f, grad_f, x0)
    """
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        max_iter: int = 1000,
        tol: float = 1e-6,
        verbose: int = 0,
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
    
    def minimize(
        self,
        f: Callable[[np.ndarray], float],
        grad_f: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
    ) -> OptimizationResult:
        """Minimize function with Adam."""
        x = x0.copy()
        m = np.zeros_like(x)  # first moment
        v = np.zeros_like(x)  # second moment
        
        history = {
            "f_vals": [],
            "grad_norms": [],
            "adaptive_lr": [],
            "m_norms": [0.0],  # Initial first moment norm (starts at zero)
            "v_norms": [0.0],  # Initial second moment norm (starts at zero)
        }
        
        for iteration in range(1, self.max_iter + 1):
            f_val = f(x)
            grad = grad_f(x)
            grad_norm = np.linalg.norm(grad)
            
            # Update biased moments
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
            
            # Bias correction
            m_hat = m / (1 - self.beta1 ** iteration)
            v_hat = v / (1 - self.beta2 ** iteration)
            
            # Adaptive learning rate
            adaptive_lr = self.learning_rate / (np.sqrt(v_hat) + self.epsilon)
            
            # Log
            history["f_vals"].append(float(f_val))
            history["grad_norms"].append(float(grad_norm))
            history["adaptive_lr"].append(float(np.mean(adaptive_lr)))
            history["m_norms"].append(float(np.linalg.norm(m)))
            history["v_norms"].append(float(np.linalg.norm(v)))
            
            if self.verbose > 0 and iteration % self.verbose == 0:
                print(
                    f"  Iter {iteration:4d}: f={f_val:.6e}, "
                    f"|grad|={grad_norm:.6e}, lr={np.mean(adaptive_lr):.6e}"
                )
            
            # Check convergence
            if grad_norm < self.tol:
                if self.verbose > 0:
                    print(f"  Converged at iteration {iteration}")
                return OptimizationResult(
                    x_final=x,
                    f_final=float(f_val),
                    history=history,
                    converged=True,
                    n_iterations=iteration,
                )
            
            # Update step
            x = x - adaptive_lr * m_hat
        
        # Max iterations
        f_final = f(x)
        history["f_vals"].append(float(f_final))
        
        return OptimizationResult(
            x_final=x,
            f_final=float(f_final),
            history=history,
            converged=False,
            n_iterations=self.max_iter,
        )


# CLI for demo
app = typer.Typer()


@app.callback(invoke_without_command=True)
def run_optim_demo(
    seed: int = typer.Option(42, help="Random seed"),
    condition_number: float = typer.Option(100.0, help="Condition number for quadratic"),
) -> None:
    """
    Run optimizer comparison demo.
    
    Compares GD, Momentum, and Adam on a quadratic objective
    with specified condition number.
    """
    from pathlib import Path
    import matplotlib.pyplot as plt
    from modules._import_helper import safe_import_from
    
    set_seed = safe_import_from('00_repo_standards.src.mlphys_core', 'set_seed')
    
    set_seed(seed)
    typer.echo(f"Running optimizer demo with kappa={condition_number}")
    
    # Create ill-conditioned quadratic
    # A = Q diag([1, kappa]) Q^T
    n = 2
    kappa = condition_number
    eigs = np.array([kappa, 1.0])
    Q = np.eye(n)
    A = Q @ np.diag(eigs) @ Q.T
    
    def f(x):
        return 0.5 * x.T @ A @ x
    
    def grad_f(x):
        return A @ x
    
    x0 = np.array([1.0, 1.0])
    
    # Run optimizers
    typer.echo("\nGradient Descent:")
    gd = GradientDescent(learning_rate=0.1, max_iter=500, verbose=100)
    result_gd = gd.minimize(f, grad_f, x0)
    typer.echo(f"  {result_gd}")
    
    typer.echo("\nMomentum:")
    momentum = MomentumOptimizer(learning_rate=0.1, momentum=0.9, max_iter=500, verbose=100)
    result_momentum = momentum.minimize(f, grad_f, x0)
    typer.echo(f"  {result_momentum}")
    
    typer.echo("\nAdam:")
    adam = AdamOptimizer(learning_rate=0.1, max_iter=500, verbose=100)
    result_adam = adam.minimize(f, grad_f, x0)
    typer.echo(f"  {result_adam}")
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.semilogy(result_gd.history["f_vals"], label="GD", linewidth=2)
    plt.semilogy(result_momentum.history["f_vals"], label="Momentum", linewidth=2)
    plt.semilogy(result_adam.history["f_vals"], label="Adam", linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Objective Value (log scale)")
    plt.title(f"Optimizer Convergence (κ={kappa:.1f})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save
    output_dir = Path("modules/01_numerical_toolbox/reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"convergence_kappa_{int(kappa)}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    typer.secho(f"\n✓ Plot saved to {output_path}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()

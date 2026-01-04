"""Linear algebra utilities for ML: PCA, SVD, conditioning analysis.

This module provides implementations of fundamental linear algebra operations
with emphasis on numerical stability and geometric interpretation.
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
import typer
import matplotlib.pyplot as plt
from pathlib import Path


@dataclass
class PCAResult:
    """Container for PCA analysis results."""
    
    components: np.ndarray  # Principal components (eigenvectors)
    explained_variance: np.ndarray  # Variance explained by each PC
    explained_variance_ratio: np.ndarray  # Fraction of total variance
    singular_values: np.ndarray  # Singular values from SVD
    mean: np.ndarray  # Data mean (for reconstruction)
    
    def transform(self, X: np.ndarray, n_components: Optional[int] = None) -> np.ndarray:
        """Project data onto principal components."""
        X_centered = X - self.mean
        n_comp = n_components if n_components is not None else self.components.shape[1]
        return X_centered @ self.components[:, :n_comp]
    
    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Reconstruct data from principal component scores."""
        n_comp = Z.shape[1]
        return Z @ self.components[:, :n_comp].T + self.mean
    
    def reconstruction_error(self, X: np.ndarray, n_components: int) -> float:
        """Compute reconstruction error with n_components."""
        Z = self.transform(X, n_components)
        X_reconstructed = self.inverse_transform(Z)
        return np.sum((X - X_reconstructed) ** 2)


def pca_via_svd(
    X: np.ndarray,
    n_components: Optional[int] = None,
    center: bool = True,
) -> PCAResult:
    """
    Perform PCA using Singular Value Decomposition.
    
    More numerically stable than eigendecomposition of covariance matrix.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
        n_components: Number of components to retain (default: all)
        center: Whether to center data (subtract mean)
        
    Returns:
        PCAResult object with components and variance info
        
    Mathematical details:
        1. Center data: X_centered = X - mean(X)
        2. Compute SVD: X_centered = U Sigma V^T
        3. Principal components = V (right singular vectors)
        4. Explained variance = Sigma^2 / (n_samples - 1)
    
    Example:
        >>> X = np.random.randn(100, 5)
        >>> pca_result = pca_via_svd(X, n_components=3)
        >>> Z = pca_result.transform(X)  # Project to 3D
        >>> X_reconstructed = pca_result.inverse_transform(Z)
    """
    n_samples, n_features = X.shape
    
    # Center data
    mean = np.mean(X, axis=0) if center else np.zeros(n_features)
    X_centered = X - mean
    
    # Compute SVD
    # Note: full_matrices=False for efficiency when n_samples >> n_features
    U, singular_values, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # Components are rows of Vt (columns of V)
    components = Vt.T
    
    # Explained variance
    explained_variance = (singular_values ** 2) / (n_samples - 1)
    total_variance = np.sum(explained_variance)
    explained_variance_ratio = explained_variance / total_variance
    
    # Keep only n_components
    if n_components is not None:
        components = components[:, :n_components]
        explained_variance = explained_variance[:n_components]
        explained_variance_ratio = explained_variance_ratio[:n_components]
        singular_values = singular_values[:n_components]
    
    return PCAResult(
        components=components,
        explained_variance=explained_variance,
        explained_variance_ratio=explained_variance_ratio,
        singular_values=singular_values,
        mean=mean,
    )


def condition_number(A: np.ndarray) -> float:
    """
    Compute condition number of matrix A.
    
    kappa(A) = sigma_max / sigma_min
    
    Args:
        A: Matrix (2D numpy array)
        
    Returns:
        Condition number (float)
        
    Interpretation:
        - kappa ≈ 1: Well-conditioned
        - kappa >> 1: Ill-conditioned (sensitive to perturbations)
    
    Example:
        >>> A = np.diag([100, 1])  # Elongated ellipse
        >>> kappa = condition_number(A)  # Should be 100
    """
    singular_values = np.linalg.svd(A, compute_uv=False)
    return float(singular_values[0] / singular_values[-1])


def ridge_regularization(
    X: np.ndarray,
    y: np.ndarray,
    lambda_: float,
) -> Tuple[np.ndarray, float, float]:
    """
    Solve ridge regression: min ||Xw - y||^2 + lambda ||w||^2
    
    Closed-form solution: w = (X^T X + lambda I)^{-1} X^T y
    
    Args:
        X: Design matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        lambda_: Regularization strength
        
    Returns:
        Tuple of (weights, condition_number_before, condition_number_after)
        
    Effect of ridge:
        - Adds lambda to all eigenvalues of X^T X
        - Improves condition number
        - Trades bias for reduced variance
    
    Example:
        >>> X = np.random.randn(100, 50)  # Possibly ill-conditioned
        >>> y = np.random.randn(100)
        >>> w, kappa_before, kappa_after = ridge_regularization(X, y, lambda_=0.1)
        >>> print(f"Condition improved: {kappa_before:.1f} -> {kappa_after:.1f}")
    """
    n_features = X.shape[1]
    
    # Compute X^T X
    XtX = X.T @ X
    kappa_before = condition_number(XtX)
    
    # Add ridge regularization
    XtX_ridge = XtX + lambda_ * np.eye(n_features)
    kappa_after = condition_number(XtX_ridge)
    
    # Solve
    weights = np.linalg.solve(XtX_ridge, X.T @ y)
    
    return weights, kappa_before, kappa_after


def demonstrate_ill_conditioning(
    kappa_target: float = 1000.0,
    n_samples: int = 100,
    n_features: int = 50,
    seed: int = 42,
) -> dict:
    """
    Demonstrate effects of ill-conditioning on linear regression.
    
    Creates an ill-conditioned design matrix and shows:
    - How condition number affects solution stability
    - Ridge regularization improvement
    
    Args:
        kappa_target: Target condition number
        n_samples: Number of samples
        n_features: Number of features
        seed: Random seed
        
    Returns:
        Dictionary with condition numbers and solution norms
    """
    np.random.seed(seed)
    
    # Create ill-conditioned matrix
    # Generate random orthogonal matrix
    Q, _ = np.linalg.qr(np.random.randn(n_features, n_features))
    
    # Create singular values with specified condition number
    # Note: κ(X^T X) = κ(X)^2, so we need κ(X) = sqrt(kappa_target)
    kappa_X = np.sqrt(kappa_target)
    singular_values = np.logspace(0, -np.log10(kappa_X), n_features)
    Sigma = np.diag(singular_values)
    
    # X = U Sigma V^T (we just use random U)
    U = np.random.randn(n_samples, n_features)
    U, _ = np.linalg.qr(U)
    X = U @ Sigma @ Q.T
    
    # Generate target with noise
    true_weights = np.random.randn(n_features)
    y = X @ true_weights + 0.1 * np.random.randn(n_samples)
    
    # OLS solution (potentially unstable)
    try:
        w_ols = np.linalg.lstsq(X, y, rcond=None)[0]
        ols_norm = np.linalg.norm(w_ols)
    except np.linalg.LinAlgError:
        w_ols = None
        ols_norm = np.inf
    
    # Ridge solutions with different lambdas
    lambdas = [0.01, 0.1, 1.0, 10.0]
    ridge_results = {}
    
    for lambda_ in lambdas:
        w_ridge, kappa_before, kappa_after = ridge_regularization(X, y, lambda_)
        ridge_results[lambda_] = {
            "weights": w_ridge,
            "norm": np.linalg.norm(w_ridge),
            "kappa_before": kappa_before,
            "kappa_after": kappa_after,
            "train_mse": np.mean((X @ w_ridge - y) ** 2),
        }
    
    return {
        "condition_number": condition_number(X.T @ X),
        "ols_norm": ols_norm,
        "ridge_results": ridge_results,
        "true_norm": np.linalg.norm(true_weights),
    }


# CLI for demos
app = typer.Typer()


@app.callback(invoke_without_command=True)
def run_pca_demo(
    seed: int = typer.Option(42, help="Random seed"),
    n_samples: int = typer.Option(200, help="Number of samples"),
    n_features: int = typer.Option(10, help="Number of features"),
) -> None:
    """
    Run PCA demonstration with synthetic correlated data.
    
    Generates data, performs PCA, and creates visualizations.
    """
    from modules._import_helper import safe_import_from
    
    set_seed = safe_import_from('00_repo_standards.src.mlphys_core', 'set_seed')
    
    set_seed(seed)
    typer.echo(f"Running PCA demo with {n_samples} samples, {n_features} features")
    
    # Generate correlated data
    # Create covariance matrix with decaying eigenvalues
    eigenvalues = np.exp(-np.arange(n_features) / 2)
    Q = np.linalg.qr(np.random.randn(n_features, n_features))[0]
    Sigma = Q @ np.diag(eigenvalues) @ Q.T
    
    X = np.random.multivariate_normal(np.zeros(n_features), Sigma, size=n_samples)
    
    # Perform PCA
    pca_result = pca_via_svd(X)
    
    typer.echo("\nExplained Variance Ratio:")
    for i, ratio in enumerate(pca_result.explained_variance_ratio[:5]):
        typer.echo(f"  PC{i+1}: {ratio:.4f} ({ratio*100:.2f}%)")
    
    cumulative_var = np.cumsum(pca_result.explained_variance_ratio)
    n_95 = np.argmax(cumulative_var >= 0.95) + 1
    typer.echo(f"\nComponents needed for 95% variance: {n_95}")
    
    # Create visualizations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scree plot
    ax = axes[0]
    ax.plot(range(1, len(pca_result.explained_variance_ratio) + 1),
            pca_result.explained_variance_ratio, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("Scree Plot")
    ax.grid(True, alpha=0.3)
    
    # Cumulative variance
    ax = axes[1]
    ax.plot(range(1, len(cumulative_var) + 1),
            cumulative_var, 'ro-', linewidth=2, markersize=8)
    ax.axhline(0.95, color='k', linestyle='--', label='95% threshold')
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("Cumulative Variance Explained")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_dir = Path("modules/01_numerical_toolbox/reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "pca_variance_explained.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    typer.secho(f"\n✓ Plot saved to {output_path}", fg=typer.colors.GREEN)
    
    # Test reconstruction
    for n_comp in [2, 5, n_features]:
        Z = pca_result.transform(X, n_components=n_comp)
        X_reconstructed = pca_result.inverse_transform(Z)
        error = np.mean((X - X_reconstructed) ** 2)
        typer.echo(f"Reconstruction error ({n_comp} components): {error:.6f}")


if __name__ == "__main__":
    app()

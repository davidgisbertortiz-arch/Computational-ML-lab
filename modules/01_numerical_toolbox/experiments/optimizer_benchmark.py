"""Optimizer benchmark experiment with MLflow tracking.

Compares GD, Momentum, and Adam on:
  (a) Quadratic bowl with varying condition numbers
  (b) Logistic regression on a small synthetic dataset

Logs iterations, loss, gradient norm, wall time.
Produces summary table and convergence plots.

Usage:
    python -m modules.01_numerical_toolbox.experiments.optimizer_benchmark
    
    # With custom settings
    python -m modules.01_numerical_toolbox.experiments.optimizer_benchmark \
        --max-iter 500 --seed 42 --mlflow-experiment "optim-benchmark"
"""

import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
import mlflow
import typer
from modules._import_helper import safe_import_from

# Local imports using Python 3.12+ workaround
(GradientDescent, MomentumOptimizer, AdamOptimizer, OptimizationResult) = safe_import_from(
    '01_numerical_toolbox.src.optimizers_from_scratch',
    'GradientDescent', 'MomentumOptimizer', 'AdamOptimizer', 'OptimizationResult'
)
create_quadratic_bowl = safe_import_from('01_numerical_toolbox.src.toy_problems', 'create_quadratic_bowl')
set_seed = safe_import_from('00_repo_standards.src.mlphys_core', 'set_seed')


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Single benchmark run result."""
    optimizer: str
    problem: str
    iterations: int
    final_loss: float
    final_grad_norm: float
    wall_time_sec: float
    converged: bool
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LogisticRegressionProblem:
    """Binary logistic regression: min -log-likelihood."""
    
    X: np.ndarray  # (n_samples, n_features)
    y: np.ndarray  # (n_samples,) binary {0, 1}
    reg_lambda: float = 0.01  # L2 regularization
    
    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        return np.where(
            z >= 0,
            1 / (1 + np.exp(-z)),
            np.exp(z) / (1 + np.exp(z))
        )
    
    def __call__(self, w: np.ndarray) -> float:
        """Negative log-likelihood + L2 regularization."""
        z = self.X @ w
        # Stable log-loss computation
        loss = -np.mean(
            self.y * z - np.logaddexp(0, z)
        )
        # L2 regularization
        loss += 0.5 * self.reg_lambda * np.sum(w ** 2)
        return loss
    
    def gradient(self, w: np.ndarray) -> np.ndarray:
        """Gradient of loss w.r.t. weights."""
        p = self.sigmoid(self.X @ w)
        grad = self.X.T @ (p - self.y) / len(self.y)
        grad += self.reg_lambda * w
        return grad


def create_logistic_problem(
    n_samples: int = 200,
    n_features: int = 10,
    separation: float = 1.5,
    seed: int = 42,
) -> LogisticRegressionProblem:
    """
    Create a synthetic binary classification problem.
    
    Args:
        n_samples: Total samples (split 50/50 between classes)
        n_features: Feature dimensionality
        separation: Distance between class centers
        seed: Random seed
    
    Returns:
        LogisticRegressionProblem instance
    """
    np.random.seed(seed)
    
    n_per_class = n_samples // 2
    
    # Class 0: centered at origin
    X0 = np.random.randn(n_per_class, n_features)
    
    # Class 1: shifted along first dimension
    X1 = np.random.randn(n_per_class, n_features)
    X1[:, 0] += separation
    
    X = np.vstack([X0, X1])
    y = np.array([0] * n_per_class + [1] * n_per_class, dtype=float)
    
    # Shuffle
    perm = np.random.permutation(n_samples)
    X, y = X[perm], y[perm]
    
    return LogisticRegressionProblem(X=X, y=y)


# -----------------------------------------------------------------------------
# Benchmark runner
# -----------------------------------------------------------------------------

def run_single_benchmark(
    optimizer_name: str,
    optimizer,
    f: Callable,
    grad_f: Callable,
    x0: np.ndarray,
    problem_name: str,
) -> BenchmarkResult:
    """Run a single optimizer benchmark."""
    
    start_time = time.perf_counter()
    result: OptimizationResult = optimizer.minimize(f, grad_f, x0.copy())
    wall_time = time.perf_counter() - start_time
    
    # Final gradient norm
    final_grad_norm = float(np.linalg.norm(grad_f(result.x_final)))
    
    return BenchmarkResult(
        optimizer=optimizer_name,
        problem=problem_name,
        iterations=result.n_iterations,
        final_loss=float(result.f_final),
        final_grad_norm=final_grad_norm,
        wall_time_sec=wall_time,
        converged=result.converged,
    )


def run_benchmark_suite(
    max_iter: int = 300,
    seed: int = 42,
) -> tuple[list[BenchmarkResult], dict]:
    """
    Run full benchmark suite.
    
    Returns:
        Tuple of (results_list, convergence_histories)
    """
    set_seed(seed)
    
    results = []
    histories = {}
    
    # Define optimizers
    optimizers = {
        "GD": GradientDescent(learning_rate=0.1, max_iter=max_iter, tol=1e-8),
        "Momentum": MomentumOptimizer(learning_rate=0.1, momentum=0.9, max_iter=max_iter, tol=1e-8),
        "Adam": AdamOptimizer(learning_rate=0.1, max_iter=max_iter, tol=1e-8),
    }
    
    # -------------------------------------------------------------------------
    # Problem 1: Quadratic bowl (well-conditioned)
    # -------------------------------------------------------------------------
    bowl_easy = create_quadratic_bowl(n_dim=10, condition_number=5.0, seed=seed)
    x0_bowl = np.ones(10) * 5.0
    
    for opt_name, opt in optimizers.items():
        result = opt.minimize(bowl_easy, bowl_easy.gradient, x0_bowl.copy())
        
        bench = BenchmarkResult(
            optimizer=opt_name,
            problem="quadratic_kappa_5",
            iterations=result.n_iterations,
            final_loss=float(result.f_final),
            final_grad_norm=float(result.history["grad_norms"][-1]),
            wall_time_sec=0.0,  # Will measure in MLflow run
            converged=result.converged,
        )
        results.append(bench)
        histories[f"quadratic_kappa_5_{opt_name}"] = result.history["f_vals"]
    
    # -------------------------------------------------------------------------
    # Problem 2: Quadratic bowl (ill-conditioned)
    # -------------------------------------------------------------------------
    bowl_hard = create_quadratic_bowl(n_dim=10, condition_number=100.0, seed=seed)
    
    optimizers_hard = {
        "GD": GradientDescent(learning_rate=0.01, max_iter=max_iter, tol=1e-8),
        "Momentum": MomentumOptimizer(learning_rate=0.01, momentum=0.9, max_iter=max_iter, tol=1e-8),
        "Adam": AdamOptimizer(learning_rate=0.1, max_iter=max_iter, tol=1e-8),
    }
    
    for opt_name, opt in optimizers_hard.items():
        result = opt.minimize(bowl_hard, bowl_hard.gradient, x0_bowl.copy())
        
        bench = BenchmarkResult(
            optimizer=opt_name,
            problem="quadratic_kappa_100",
            iterations=result.n_iterations,
            final_loss=float(result.f_final),
            final_grad_norm=float(result.history["grad_norms"][-1]),
            wall_time_sec=0.0,
            converged=result.converged,
        )
        results.append(bench)
        histories[f"quadratic_kappa_100_{opt_name}"] = result.history["f_vals"]
    
    # -------------------------------------------------------------------------
    # Problem 3: Logistic regression
    # -------------------------------------------------------------------------
    logreg = create_logistic_problem(n_samples=200, n_features=10, seed=seed)
    x0_logreg = np.zeros(10)
    
    optimizers_logreg = {
        "GD": GradientDescent(learning_rate=0.5, max_iter=max_iter, tol=1e-8),
        "Momentum": MomentumOptimizer(learning_rate=0.5, momentum=0.9, max_iter=max_iter, tol=1e-8),
        "Adam": AdamOptimizer(learning_rate=0.1, max_iter=max_iter, tol=1e-8),
    }
    
    for opt_name, opt in optimizers_logreg.items():
        start_time = time.perf_counter()
        result = opt.minimize(logreg, logreg.gradient, x0_logreg.copy())
        wall_time = time.perf_counter() - start_time
        
        bench = BenchmarkResult(
            optimizer=opt_name,
            problem="logistic_regression",
            iterations=result.n_iterations,
            final_loss=float(result.f_final),
            final_grad_norm=float(result.history["grad_norms"][-1]),
            wall_time_sec=wall_time,
            converged=result.converged,
        )
        results.append(bench)
        histories[f"logistic_regression_{opt_name}"] = result.history["f_vals"]
    
    return results, histories


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------

def create_summary_table(results: list[BenchmarkResult]) -> str:
    """Create a formatted summary table."""
    
    header = f"{'Problem':<25} {'Optimizer':<12} {'Iters':>8} {'Loss':>12} {'|∇f|':>12} {'Time(s)':>10} {'Conv':>6}"
    sep = "-" * len(header)
    
    lines = [sep, header, sep]
    
    for r in results:
        conv_str = "✓" if r.converged else "✗"
        line = (
            f"{r.problem:<25} {r.optimizer:<12} {r.iterations:>8} "
            f"{r.final_loss:>12.6f} {r.final_grad_norm:>12.2e} "
            f"{r.wall_time_sec:>10.4f} {conv_str:>6}"
        )
        lines.append(line)
    
    lines.append(sep)
    return "\n".join(lines)


def create_convergence_plots(
    histories: dict,
    output_dir: Path,
) -> Path:
    """Create convergence plots for all problems."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    problems = [
        ("quadratic_kappa_5", "Quadratic (κ=5)"),
        ("quadratic_kappa_100", "Quadratic (κ=100)"),
        ("logistic_regression", "Logistic Regression"),
    ]
    
    colors = {"GD": "C0", "Momentum": "C1", "Adam": "C2"}
    
    for ax, (prob_key, prob_title) in zip(axes, problems):
        for opt_name in ["GD", "Momentum", "Adam"]:
            key = f"{prob_key}_{opt_name}"
            if key in histories:
                f_vals = np.array(histories[key])
                # Plot loss - minimum (use log scale)
                min_val = f_vals[-1]
                ax.semilogy(
                    f_vals - min_val + 1e-10,
                    label=opt_name,
                    color=colors[opt_name],
                    linewidth=2,
                )
        
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss - min (log scale)")
        ax.set_title(prob_title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / "convergence_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    return output_path


# -----------------------------------------------------------------------------
# MLflow integration
# -----------------------------------------------------------------------------

def log_to_mlflow(
    results: list[BenchmarkResult],
    histories: dict,
    output_dir: Path,
    experiment_name: str,
) -> str:
    """Log benchmark results to MLflow."""
    
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name="optimizer_benchmark"):
        # Log parameters
        mlflow.log_param("n_problems", 3)
        mlflow.log_param("n_optimizers", 3)
        mlflow.log_param("seed", 42)
        
        # Log summary metrics
        for r in results:
            prefix = f"{r.problem}_{r.optimizer}"
            mlflow.log_metric(f"{prefix}_iterations", r.iterations)
            mlflow.log_metric(f"{prefix}_final_loss", r.final_loss)
            mlflow.log_metric(f"{prefix}_grad_norm", r.final_grad_norm)
            mlflow.log_metric(f"{prefix}_wall_time", r.wall_time_sec)
        
        # Save and log convergence plot
        plot_path = create_convergence_plots(histories, output_dir)
        mlflow.log_artifact(str(plot_path))
        
        # Save and log summary table
        table_str = create_summary_table(results)
        table_path = output_dir / "summary_table.txt"
        table_path.write_text(table_str)
        mlflow.log_artifact(str(table_path))
        
        # Save raw results as JSON
        results_json = [r.to_dict() for r in results]
        json_path = output_dir / "benchmark_results.json"
        json_path.write_text(json.dumps(results_json, indent=2))
        mlflow.log_artifact(str(json_path))
        
        run_id = mlflow.active_run().info.run_id
    
    return run_id


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

app = typer.Typer(help="Optimizer benchmark experiment")


@app.command()
def run(
    max_iter: int = typer.Option(300, help="Maximum iterations per optimizer"),
    seed: int = typer.Option(42, help="Random seed"),
    mlflow_experiment: str = typer.Option(
        "numerical-toolbox-benchmark",
        help="MLflow experiment name",
    ),
    output_dir: str = typer.Option(
        "modules/01_numerical_toolbox/reports/benchmark",
        help="Output directory for artifacts",
    ),
    no_mlflow: bool = typer.Option(False, help="Skip MLflow logging"),
) -> None:
    """
    Run optimizer benchmark comparing GD, Momentum, and Adam.
    
    Problems:
    - Quadratic bowl (κ=5, well-conditioned)
    - Quadratic bowl (κ=100, ill-conditioned)  
    - Logistic regression (200 samples, 10 features)
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    typer.echo("=" * 60)
    typer.echo("Optimizer Benchmark Experiment")
    typer.echo("=" * 60)
    typer.echo(f"Max iterations: {max_iter}")
    typer.echo(f"Seed: {seed}")
    typer.echo(f"Output: {output_path}")
    typer.echo("")
    
    # Run benchmarks
    typer.echo("Running benchmarks...")
    results, histories = run_benchmark_suite(max_iter=max_iter, seed=seed)
    
    # Print summary table
    table = create_summary_table(results)
    typer.echo("\n" + table + "\n")
    
    # Create plots
    plot_path = create_convergence_plots(histories, output_path)
    typer.secho(f"✓ Convergence plot saved: {plot_path}", fg=typer.colors.GREEN)
    
    # Log to MLflow
    if not no_mlflow:
        typer.echo("\nLogging to MLflow...")
        try:
            run_id = log_to_mlflow(results, histories, output_path, mlflow_experiment)
            typer.secho(f"✓ MLflow run ID: {run_id}", fg=typer.colors.GREEN)
            typer.echo(f"  View: mlflow ui --backend-store-uri mlruns")
        except Exception as e:
            typer.secho(f"⚠ MLflow logging failed: {e}", fg=typer.colors.YELLOW)
    
    # Save summary
    table_path = output_path / "summary_table.txt"
    table_path.write_text(table)
    typer.secho(f"✓ Summary table saved: {table_path}", fg=typer.colors.GREEN)
    
    typer.echo("\n" + "=" * 60)
    typer.secho("Benchmark complete!", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()

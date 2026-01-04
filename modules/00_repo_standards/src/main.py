"""CLI entry point for Module 00 experiments."""

import typer
import numpy as np
from pathlib import Path
import importlib

# Workaround for Python 3.12+ octal literal parsing
_core = importlib.import_module('modules.00_repo_standards.src.core')
_utils = importlib.import_module('modules.00_repo_standards.src.utils')

gradient_descent = _core.gradient_descent
set_seed = _utils.set_seed
load_config = _utils.load_config
log_experiment = _utils.log_experiment
save_results = _utils.save_results

app = typer.Typer(help="Module 00: Repository Standards - Example CLI")


@app.command()
def train(
    config: Path = typer.Option(
        "modules/00_repo_standards/configs/example.yaml",
        help="Path to config file",
    ),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
    verbose: bool = typer.Option(False, help="Print detailed progress"),
) -> None:
    """
    Run gradient descent experiment on quadratic function.
    
    Demonstrates:
    - Loading configuration
    - Setting seeds
    - Running optimization
    - Logging results
    """
    # Load config
    cfg = load_config(config)
    typer.echo(f"Loaded config from {config}")
    
    # Set seed for reproducibility
    set_seed(seed)
    typer.echo(f"Set random seed: {seed}")
    
    # Define objective: f(x) = x^T x (sum of squares)
    def quadratic_grad(x: np.ndarray) -> np.ndarray:
        return 2 * x
    
    # Initial point (use simple dimension)
    dim = 10
    lr = 0.01
    max_iter = 1000
    tol = 1e-6
    
    x0 = np.random.randn(dim)
    typer.echo(f"Initial point: {x0[:3]}... (dim={dim})")
    
    # Run gradient descent
    typer.echo("Running gradient descent...")
    x_opt, history = gradient_descent(
        x0=x0,
        grad_fn=quadratic_grad,
        lr=lr,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
    )
    
    # Compute metrics
    final_norm = float(np.linalg.norm(x_opt))
    iterations = len(history)
    converged = history[-1] < tol
    
    # Log results
    params = {
        "seed": seed,
        "dim": dim,
        "lr": lr,
        "max_iter": max_iter,
        "tol": tol,
    }
    
    metrics = {
        "final_norm": final_norm,
        "iterations": iterations,
        "converged": converged,
        "initial_grad_norm": history[0],
        "final_grad_norm": history[-1],
    }
    
    log_experiment(
        experiment_name="gradient_descent_quadratic",
        params=params,
        metrics=metrics,
    )
    
    # Save results
    output_dir = Path("modules/00_repo_standards/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "params": params,
        "metrics": metrics,
        "optimal_point": x_opt.tolist(),
        "history": history,
    }
    
    save_results(results, output_dir / "experiment_results.yaml")
    
    if converged:
        typer.secho("✓ Optimization converged!", fg=typer.colors.GREEN)
    else:
        typer.secho("✗ Did not converge", fg=typer.colors.YELLOW)


@app.command()
def test_reproducibility(
    seed: int = typer.Option(42, help="Random seed"),
    n_runs: int = typer.Option(3, help="Number of runs to test"),
) -> None:
    """
    Test that experiments are reproducible with the same seed.
    
    Runs the same experiment multiple times and checks that results match.
    """
    typer.echo(f"Testing reproducibility with {n_runs} runs (seed={seed})")
    
    results = []
    
    for i in range(n_runs):
        set_seed(seed)
        x0 = np.random.randn(5)
        
        x_opt, history = gradient_descent(
            x0=x0,
            grad_fn=lambda x: 2 * x,
            lr=0.1,
            max_iter=100,
            tol=1e-6,
        )
        
        results.append({
            "x0": x0.copy(),
            "x_opt": x_opt.copy(),
            "final_grad_norm": history[-1],
        })
    
    # Check all runs match
    all_match = True
    for i in range(1, n_runs):
        x0_match = np.allclose(results[i]["x0"], results[0]["x0"])
        opt_match = np.allclose(results[i]["x_opt"], results[0]["x_opt"])
        
        if not (x0_match and opt_match):
            all_match = False
            typer.secho(f"✗ Run {i} does not match run 0", fg=typer.colors.RED)
    
    if all_match:
        typer.secho(
            f"✓ All {n_runs} runs produced identical results!",
            fg=typer.colors.GREEN,
        )
    else:
        typer.secho(
            "✗ Reproducibility test failed",
            fg=typer.colors.RED,
        )


if __name__ == "__main__":
    app()

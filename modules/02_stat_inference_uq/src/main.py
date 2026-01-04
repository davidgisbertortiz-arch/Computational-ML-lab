"""CLI entry point for Module 02: Statistical Inference & UQ."""

import typer
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from modules._import_helper import safe_import_from

# Python 3.12+ workaround for numeric module names
set_seed, get_rng = safe_import_from(
    '00_repo_standards.src.mlphys_core.seeding',
    'set_seed', 'get_rng'
)
BayesianLinearRegression, posterior_predictive = safe_import_from(
    '02_stat_inference_uq.src.bayesian_regression',
    'BayesianLinearRegression', 'posterior_predictive'
)
reliability_diagram, expected_calibration_error, TemperatureScaling = safe_import_from(
    '02_stat_inference_uq.src.calibration',
    'reliability_diagram', 'expected_calibration_error', 'TemperatureScaling'
)
MetropolisHastings, MCMCDiagnostics = safe_import_from(
    '02_stat_inference_uq.src.mcmc_basics',
    'MetropolisHastings', 'MCMCDiagnostics'
)

app = typer.Typer(help="Module 02: Statistical Inference & Uncertainty Quantification")


@app.command()
def run_bayes_demo(
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
    n_samples: int = typer.Option(100, help="Number of training samples"),
    noise_std: float = typer.Option(0.3, help="Observation noise std"),
    output_dir: Path = typer.Option(Path("modules/02_stat_inference_uq/reports"), help="Output directory"),
) -> None:
    """
    Run Bayesian regression demo comparing posterior intervals vs frequentist CI.
    
    Generates synthetic linear data and fits both Bayesian and frequentist models,
    visualizing prediction uncertainty.
    """
    typer.echo(f"🔬 Running Bayesian regression demo (seed={seed})")
    
    # Setup
    set_seed(seed)
    rng = get_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic data: y = 2x - 1 + noise
    X_train = rng.uniform(-2, 2, (n_samples, 1))
    true_weights = np.array([2.0, -1.0])  # [slope, intercept]
    y_train = X_train[:, 0] * true_weights[0] + true_weights[1] + noise_std * rng.standard_normal(n_samples)
    
    # Test points
    X_test = np.linspace(-3, 3, 100).reshape(-1, 1)
    
    # --- Bayesian Model ---
    typer.echo("  Fitting Bayesian linear regression...")
    bayes_model = BayesianLinearRegression(noise_variance=noise_std**2, fit_intercept=True)
    bayes_model.fit(X_train, y_train)
    
    y_pred_bayes, y_std_bayes = bayes_model.predict(X_test, return_std=True)
    
    # 95% credible intervals
    lower_bayes = y_pred_bayes - 1.96 * y_std_bayes
    upper_bayes = y_pred_bayes + 1.96 * y_std_bayes
    
    # --- Frequentist Model (OLS with bootstrap CI) ---
    typer.echo("  Fitting frequentist OLS with bootstrap CI...")
    from sklearn.linear_model import LinearRegression
    
    freq_model = LinearRegression()
    freq_model.fit(X_train, y_train)
    y_pred_freq = freq_model.predict(X_test)
    
    # Bootstrap for confidence intervals
    n_bootstrap = 500
    bootstrap_preds = np.zeros((n_bootstrap, len(X_test)))
    
    for i in range(n_bootstrap):
        indices = rng.integers(0, n_samples, n_samples)
        X_boot, y_boot = X_train[indices], y_train[indices]
        boot_model = LinearRegression()
        boot_model.fit(X_boot, y_boot)
        bootstrap_preds[i] = boot_model.predict(X_test)
    
    lower_freq = np.percentile(bootstrap_preds, 2.5, axis=0)
    upper_freq = np.percentile(bootstrap_preds, 97.5, axis=0)
    
    # --- Visualization ---
    typer.echo("  Creating visualizations...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bayesian plot
    ax = axes[0]
    ax.scatter(X_train, y_train, alpha=0.5, color="black", label="Training data", s=30)
    ax.plot(X_test, y_pred_bayes, "b-", label="Posterior mean", linewidth=2)
    ax.fill_between(X_test.flatten(), lower_bayes, upper_bayes, alpha=0.3, color="blue", label="95% credible interval")
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_title("Bayesian Linear Regression", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Frequentist plot
    ax = axes[1]
    ax.scatter(X_train, y_train, alpha=0.5, color="black", label="Training data", s=30)
    ax.plot(X_test, y_pred_freq, "r-", label="OLS fit", linewidth=2)
    ax.fill_between(X_test.flatten(), lower_freq, upper_freq, alpha=0.3, color="red", label="95% bootstrap CI")
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_title("Frequentist OLS (Bootstrap CI)", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    
    fig.tight_layout()
    output_path = output_dir / "bayes_vs_frequentist.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    typer.echo(f"  ✅ Plot saved to {output_path}")
    
    # Print diagnostics
    typer.echo(f"\n📊 Diagnostics:")
    typer.echo(f"  Posterior mean weights: {bayes_model.posterior_mean_}")
    typer.echo(f"  True weights: {np.array([true_weights[1], true_weights[0]])}  (intercept, slope)")
    typer.echo(f"  Frequentist weights: intercept={freq_model.intercept_:.3f}, slope={freq_model.coef_[0]:.3f}")
    typer.echo(f"  Mean Bayesian CI width: {np.mean(upper_bayes - lower_bayes):.3f}")
    typer.echo(f"  Mean frequentist CI width: {np.mean(upper_freq - lower_freq):.3f}")


@app.command()
def run_calibration_demo(
    seed: int = typer.Option(42, help="Random seed"),
    n_samples: int = typer.Option(500, help="Number of samples"),
    output_dir: Path = typer.Option(Path("modules/02_stat_inference_uq/reports"), help="Output directory"),
) -> None:
    """
    Run calibration demo showing miscalibrated classifier and improvement with temperature scaling.
    """
    typer.echo(f"📏 Running calibration demo (seed={seed})")
    
    set_seed(seed)
    rng = get_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic miscalibrated predictions
    # True accuracy is ~70%, but model is overconfident
    y_true = rng.integers(0, 2, n_samples)
    
    # Simulated logits (overconfident)
    logits = np.where(y_true == 1, 2.5, -2.5) + rng.normal(0, 1.0, n_samples)
    
    # Randomly flip 30% to create 70% accuracy
    flip_mask = rng.random(n_samples) < 0.3
    y_true[flip_mask] = 1 - y_true[flip_mask]
    
    # Split into val and test
    n_val = int(0.6 * n_samples)
    logits_val, logits_test = logits[:n_val], logits[n_val:]
    y_val, y_test = y_true[:n_val], y_true[n_val:]
    
    # Uncalibrated probabilities
    probs_uncalib = 1 / (1 + np.exp(-logits_test))
    ece_before = expected_calibration_error(y_test, probs_uncalib, n_bins=10)
    
    # Apply temperature scaling
    typer.echo("  Fitting temperature scaling on validation set...")
    ts = TemperatureScaling(n_classes=2, max_iter=150)
    ts.fit(logits_val, y_val)
    
    # Calibrated probabilities
    probs_calib = ts.predict_proba(logits_test)
    ece_after = expected_calibration_error(y_test, probs_calib, n_bins=10)
    
    typer.echo(f"  Learned temperature: T={ts.temperature_:.3f}")
    typer.echo(f"  ECE before: {ece_before:.4f}")
    typer.echo(f"  ECE after:  {ece_after:.4f}")
    typer.echo(f"  Improvement: {(ece_before - ece_after) / ece_before * 100:.1f}%")
    
    # Visualization
    typer.echo("  Creating reliability diagrams...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    reliability_diagram(y_test, probs_uncalib, n_bins=10, ax=axes[0])
    axes[0].set_title(f"Before Calibration (ECE={ece_before:.4f})", fontweight="bold")
    
    reliability_diagram(y_test, probs_calib, n_bins=10, ax=axes[1])
    axes[1].set_title(f"After Temperature Scaling (ECE={ece_after:.4f})", fontweight="bold")
    
    fig.tight_layout()
    output_path = output_dir / "calibration_before_after.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    typer.echo(f"  ✅ Plots saved to {output_path}")


@app.command()
def run_mcmc_demo(
    seed: int = typer.Option(42, help="Random seed"),
    n_samples: int = typer.Option(10000, help="Number of MCMC samples"),
    output_dir: Path = typer.Option(Path("modules/02_stat_inference_uq/reports"), help="Output directory"),
) -> None:
    """
    Run MCMC demo sampling from 2D Gaussian and visualizing diagnostics.
    """
    typer.echo(f"⛓️  Running MCMC demo (seed={seed})")
    
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Target: 2D Gaussian with correlation
    rho = 0.7
    cov = np.array([[1, rho], [rho, 1]])
    cov_inv = np.linalg.inv(cov)
    
    def log_prob(x):
        return -0.5 * x @ cov_inv @ x
    
    # Run sampler
    typer.echo(f"  Sampling {n_samples} points...")
    sampler = MetropolisHastings(
        log_prob_fn=log_prob,
        proposal_std=1.2,
        n_samples=n_samples,
        n_burn=2000,
        random_state=seed,
    )
    
    samples = sampler.sample(x0=np.zeros(2), verbose=False)
    
    diagnostics = sampler.get_diagnostics()
    
    typer.echo(f"\n📊 MCMC Diagnostics:")
    typer.echo(f"  Acceptance rate: {diagnostics['acceptance_rate']:.2%}")
    typer.echo(f"  ESS: {diagnostics['ess']}")
    typer.echo(f"  Integrated autocorr time: {diagnostics['autocorr_time']}")
    typer.echo(f"  Sample mean: {diagnostics['mean']}")
    typer.echo(f"  Sample std: {diagnostics['std']}")
    
    # Visualizations
    typer.echo("  Creating visualizations...")
    diag = MCMCDiagnostics(samples)
    
    # Trace plot
    fig_trace = diag.trace_plot()
    fig_trace.savefig(output_dir / "mcmc_trace.png", dpi=150, bbox_inches="tight")
    plt.close(fig_trace)
    
    # Marginals
    fig_marg = diag.marginal_histograms()
    fig_marg.savefig(output_dir / "mcmc_marginals.png", dpi=150, bbox_inches="tight")
    plt.close(fig_marg)
    
    typer.echo(f"  ✅ Plots saved to {output_dir}")


@app.command()
def run_uq_comparison(
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
    output_dir: Path = typer.Option(
        Path("modules/02_stat_inference_uq/reports/uq_classification"),
        help="Output directory for report"
    ),
) -> None:
    """
    Run UQ classification comparison experiment.
    
    Compares uncertainty quantification methods on classification:
    - Raw predict_proba (baseline)
    - Temperature scaling (post-hoc calibration)
    - Bootstrap ensembles (epistemic uncertainty)
    
    Generates report with table and diagnostic plots.
    """
    typer.echo(f"🔬 Running UQ Classification Comparison (seed={seed})")
    
    # Import and run the experiment
    from modules._import_helper import safe_import_from
    run_comparison = safe_import_from(
        '02_stat_inference_uq.experiments.uq_classification_comparison',
        'run_comparison'
    )
    
    run_comparison(seed=seed, output_dir=output_dir)
    
    typer.echo(f"\n✅ Report generated in: {output_dir}")


if __name__ == "__main__":
    app()

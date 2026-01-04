"""CLI for Module 05: Simulation & Monte Carlo."""

import typer
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

from modules._import_helper import safe_import_from

(MCIntegrator, ImportanceSampler, ControlVariates,
 RareEventEstimator, generate_brownian_motion,
 PhysicsDataGenerator) = safe_import_from(
    '05_simulation_monte_carlo.src',
    'MCIntegrator', 'ImportanceSampler', 'ControlVariates',
    'RareEventEstimator', 'generate_brownian_motion',
    'PhysicsDataGenerator'
)

app = typer.Typer(help="Monte Carlo simulation demos and experiments")


@app.command()
def run_mc_demo(
    n_samples: int = typer.Option(10000, help="Number of MC samples"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
    output_dir: Path = typer.Option("reports/mc_demo", help="Output directory for plots"),
):
    """
    Run comprehensive Monte Carlo demonstration.
    
    Generates report with:
    - Error vs N plots for MC integration
    - Confidence interval coverage analysis
    - Variance reduction comparison
    - Rare event estimation examples
    """
    typer.echo(f"🎲 Running Monte Carlo demo with {n_samples:,} samples (seed={seed})")
    
    # Setup output directory
    output_dir = Path(__file__).parent.parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ===== MC Integration =====
    typer.echo("\n📊 1. Monte Carlo Integration")
    
    # Example: ∫₀¹ e^x dx = e - 1
    def integrand(x):
        return np.exp(x)
    
    true_value = np.e - 1
    
    integrator = MCIntegrator(
        integrand=integrand,
        lower=0.0,
        upper=1.0,
        seed=seed
    )
    
    # Convergence analysis
    typer.echo("   Running convergence analysis...")
    convergence = integrator.convergence_analysis(
        sample_sizes=[100, 500, 1000, 2000, 5000, 10000, 20000],
        n_trials=50
    )
    
    # Plot error vs N
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(convergence.sample_sizes, convergence.mean_errors, 'o-', 
              label='Mean Absolute Error', linewidth=2, markersize=8)
    ax.loglog(convergence.sample_sizes, convergence.std_errors, 's-',
              label='Std Error', linewidth=2, markersize=8)
    
    # Reference line: 1/sqrt(N)
    ref = 0.5 * np.array(convergence.sample_sizes)**(-0.5)
    ax.loglog(convergence.sample_sizes, ref, 'k--', alpha=0.5, label=r'$1/\sqrt{N}$')
    
    ax.set_xlabel('Number of Samples', fontsize=12)
    ax.set_ylabel('Error', fontsize=12)
    ax.set_title('Monte Carlo Integration Convergence', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / '01_mc_convergence.png', dpi=150)
    plt.close()
    
    typer.echo(f"   ✓ Saved convergence plot to {output_dir / '01_mc_convergence.png'}")
    
    # ===== CI Coverage =====
    typer.echo("\n📈 2. Confidence Interval Coverage")
    
    n_trials = 1000
    coverage_counts = {90: 0, 95: 0, 99: 0}
    
    for trial in range(n_trials):
        integrator_tmp = MCIntegrator(
            integrand=integrand,
            lower=0.0,
            upper=1.0,
            seed=seed + trial
        )
        result = integrator_tmp.estimate(n_samples=n_samples)
        
        for alpha_pct in [90, 95, 99]:
            alpha = 1 - alpha_pct / 100
            ci_lower, ci_upper = integrator_tmp.confidence_interval(
                result.estimate, result.std_error, alpha=alpha
            )
            if ci_lower <= true_value <= ci_upper:
                coverage_counts[alpha_pct] += 1
    
    typer.echo("   Coverage results:")
    for alpha_pct, count in coverage_counts.items():
        coverage = count / n_trials
        typer.echo(f"     {alpha_pct}% CI: {coverage:.1%} (expected: {alpha_pct}%)")
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    alphas = list(coverage_counts.keys())
    observed = [coverage_counts[a] / n_trials * 100 for a in alphas]
    expected = alphas
    
    x = np.arange(len(alphas))
    width = 0.35
    
    ax.bar(x - width/2, expected, width, label='Expected', alpha=0.7)
    ax.bar(x + width/2, observed, width, label='Observed', alpha=0.7)
    
    ax.set_xlabel('Confidence Level (%)', fontsize=12)
    ax.set_ylabel('Coverage (%)', fontsize=12)
    ax.set_title('Confidence Interval Coverage Analysis', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{a}%' for a in alphas])
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / '02_ci_coverage.png', dpi=150)
    plt.close()
    
    typer.echo(f"   ✓ Saved CI coverage plot to {output_dir / '02_ci_coverage.png'}")
    
    # ===== Variance Reduction =====
    typer.echo("\n🎯 3. Variance Reduction Comparison")
    
    # Target: E[exp(-X^2)] for X ~ N(0,1)
    def target_func(x):
        return np.exp(-x**2)
    
    true_var_red = 1.0 / np.sqrt(3)
    
    # Naive MC
    rng_naive = np.random.default_rng(seed)
    samples_naive = rng_naive.standard_normal(n_samples)
    naive_est = np.mean(target_func(samples_naive))
    naive_var = np.var(target_func(samples_naive), ddof=1)
    
    # Control variates (use X^2 as control, E[X^2] = 1)
    cv = ControlVariates(
        target_func=target_func,
        control_func=lambda x: x**2,
        control_mean=1.0,
        seed=seed
    )
    cv_result = cv.estimate(n_samples=n_samples)
    
    typer.echo("   Variance reduction results:")
    typer.echo(f"     Naive MC:")
    typer.echo(f"       Estimate: {naive_est:.6f}")
    typer.echo(f"       Variance: {naive_var:.6f}")
    typer.echo(f"     Control Variates:")
    typer.echo(f"       Estimate: {cv_result.estimate:.6f}")
    typer.echo(f"       Variance: {cv_result.variance:.6f}")
    typer.echo(f"       VRF: {cv_result.variance_reduction_factor:.2f}x")
    
    # ===== Rare Events =====
    typer.echo("\n⚡ 4. Rare Event Estimation")
    
    estimator = RareEventEstimator(seed=seed)
    
    threshold = 4.0
    true_prob = 1 - 0.999968  # P(X > 4) for X ~ N(0,1)
    
    # Compare methods
    naive_rare = estimator.estimate_tail_probability(
        distribution='normal',
        threshold=threshold,
        n_samples=n_samples,
        method='naive'
    )
    
    is_rare = estimator.estimate_tail_probability(
        distribution='normal',
        threshold=threshold,
        n_samples=n_samples,
        method='importance_sampling'
    )
    
    typer.echo(f"   Estimating P(X > {threshold}) for X ~ N(0,1)")
    typer.echo(f"   True probability: {true_prob:.6e}")
    typer.echo(f"     Naive MC:")
    typer.echo(f"       Estimate: {naive_rare.probability:.6e}")
    typer.echo(f"       Rel. Error: {naive_rare.relative_error:.2%}")
    typer.echo(f"     Importance Sampling:")
    typer.echo(f"       Estimate: {is_rare.probability:.6e}")
    typer.echo(f"       Rel. Error: {is_rare.relative_error:.2%}")
    typer.echo(f"       VRF: {is_rare.variance_reduction_factor:.2f}x")
    
    typer.echo(f"\n✅ Demo complete! Results saved to {output_dir}/")
    typer.echo(f"   Generated 2 plots:")
    typer.echo(f"     - 01_mc_convergence.png")
    typer.echo(f"     - 02_ci_coverage.png")


@app.command()
def generate_physics_data(
    dataset: str = typer.Option("oscillator", help="Dataset type: oscillator, projectile, heat, pendulum"),
    n_samples: int = typer.Option(1000, help="Number of samples"),
    noise_level: float = typer.Option(0.1, help="Noise level (std dev)"),
    seed: int = typer.Option(42, help="Random seed"),
    output_file: Optional[Path] = typer.Option(None, help="Output CSV file"),
):
    """Generate physics-inspired synthetic datasets."""
    typer.echo(f"🔬 Generating {dataset} dataset with {n_samples} samples")
    
    generator = PhysicsDataGenerator(seed=seed)
    
    # Generate dataset
    dataset_map = {
        'oscillator': generator.damped_harmonic_oscillator,
        'projectile': generator.projectile_motion,
        'heat': generator.heat_diffusion_1d,
        'pendulum': generator.pendulum_energy,
    }
    
    if dataset not in dataset_map:
        typer.echo(f"❌ Unknown dataset: {dataset}", err=True)
        typer.echo(f"Available: {list(dataset_map.keys())}")
        raise typer.Exit(1)
    
    data = dataset_map[dataset](n_samples=n_samples, noise_level=noise_level)
    
    typer.echo(f"   Features shape: {data.features.shape}")
    typer.echo(f"   Targets shape: {data.targets.shape}")
    typer.echo(f"   Noise level: {data.noise_level}")
    typer.echo(f"   Description: {data.description}")
    
    # Save to CSV if requested
    if output_file:
        import pandas as pd
        df = pd.DataFrame(
            data.features,
            columns=[f'feature_{i}' for i in range(data.features.shape[1])]
        )
        df['target'] = data.targets
        df.to_csv(output_file, index=False)
        typer.echo(f"   ✓ Saved to {output_file}")


@app.command()
def generate_stochastic_process(
    process: str = typer.Option("brownian", help="Process type: brownian, ou, levy"),
    n_steps: int = typer.Option(1000, help="Number of time steps"),
    seed: int = typer.Option(42, help="Random seed"),
    output_plot: Optional[Path] = typer.Option(None, help="Output plot file"),
):
    """Generate and plot stochastic processes."""
    typer.echo(f"📈 Generating {process} motion with {n_steps} steps")
    
    # Generate process
    if process == 'brownian':
        from modules._import_helper import safe_import_from
        generate_brownian_motion = safe_import_from(
            '05_simulation_monte_carlo.src.synthetic_generators',
            'generate_brownian_motion'
        )
        t, x = generate_brownian_motion(n_steps=n_steps, seed=seed)
        title = 'Brownian Motion'
    elif process == 'ou':
        from modules._import_helper import safe_import_from
        generate_ou_process = safe_import_from(
            '05_simulation_monte_carlo.src.synthetic_generators',
            'generate_ou_process'
        )
        t, x = generate_ou_process(n_steps=n_steps, seed=seed)
        title = 'Ornstein-Uhlenbeck Process'
    elif process == 'levy':
        from modules._import_helper import safe_import_from
        generate_levy_flight = safe_import_from(
            '05_simulation_monte_carlo.src.synthetic_generators',
            'generate_levy_flight'
        )
        t, x = generate_levy_flight(n_steps=n_steps, seed=seed)
        title = 'Lévy Flight'
    else:
        typer.echo(f"❌ Unknown process: {process}", err=True)
        typer.echo("Available: brownian, ou, levy")
        raise typer.Exit(1)
    
    typer.echo(f"   Time range: [{t[0]:.2f}, {t[-1]:.2f}]")
    typer.echo(f"   Value range: [{x.min():.2f}, {x.max():.2f}]")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, x, linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Position', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_plot:
        plt.savefig(output_plot, dpi=150)
        typer.echo(f"   ✓ Saved plot to {output_plot}")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    app()

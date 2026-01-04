"""Main CLI entry point for Module 04."""

import typer
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

from modules._import_helper import safe_import_from

# Import local modules
from .config import TrackingConfig, ForecastingConfig
from .kalman import KalmanFilter, constant_velocity_model, position_observation_model
from .ekf import ExtendedKalmanFilter, pendulum_dynamics, angle_observation_model
from .particle_filter import ParticleFilter, gaussian_likelihood, create_process_noise_wrapper
from .forecasting import RollingWindowBacktest, naive_forecast, moving_average_forecast
from .hybrid_models import (
    HybridEKF, HybridEKFConfig, HybridParticleFilter,
    create_nonlinear_measurement_system, generate_training_data
)

# Import core utilities
set_seed, load_config = safe_import_from(
    '00_repo_standards.src.mlphys_core',
    'set_seed', 'load_config'
)

app = typer.Typer(help="Module 04: Time Series & State Space")


@app.command()
def run_tracking_demo(
    config_path: Path = typer.Option(
        "configs/tracking_default.yaml",
        help="Path to config file"
    ),
    seed: int = typer.Option(42, help="Random seed"),
):
    """Run state estimation tracking demonstration."""
    
    # Load config
    config = load_config(config_path, TrackingConfig)
    config.seed = seed
    set_seed(seed)
    
    print(f"=== Tracking Demo: {config.system} with {config.filter_type} ===")
    print(f"Timesteps: {config.n_timesteps}, dt: {config.dt}")
    print(f"Process noise: {config.process_noise}, Obs noise: {config.obs_noise}")
    
    # Setup output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if config.system == "constant_velocity":
        results = run_constant_velocity_tracking(config)
    elif config.system == "pendulum":
        results = run_pendulum_tracking(config)
    else:
        raise ValueError(f"Unknown system: {config.system}")
    
    # Compute RMSE
    rmse = np.sqrt(np.mean((results["true_states"] - results["estimated_states"])**2))
    print(f"\nRMSE: {rmse:.4f}")
    
    # Save results
    if config.save_plots:
        save_tracking_plots(results, output_dir, config)
        print(f"\nPlots saved to {output_dir}")
    
    # Save metrics
    metrics = {
        "rmse": float(rmse),
        "system": config.system,
        "filter": config.filter_type,
        "n_timesteps": config.n_timesteps,
    }
    
    import json
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("\n✓ Tracking demo complete!")


def run_constant_velocity_tracking(config: TrackingConfig) -> dict:
    """Run constant velocity tracking with specified filter."""
    
    rng = np.random.default_rng(config.seed)
    
    # Ground truth simulation
    true_states = []
    observations = []
    
    # Initial state: [position=0, velocity=1]
    x_true = np.array([0.0, 1.0])
    
    # Dynamics
    F, Q = constant_velocity_model(config.dt, config.process_noise)
    H, R = position_observation_model(config.obs_noise)
    
    for t in range(config.n_timesteps):
        # True dynamics with process noise
        w = rng.multivariate_normal(np.zeros(2), Q)
        x_true = F @ x_true + w
        true_states.append(x_true.copy())
        
        # Noisy observation
        v = rng.normal(0, config.obs_noise)
        z = H @ x_true + v
        observations.append(z[0])
    
    true_states = np.array(true_states)
    observations = np.array(observations)
    
    # Run filter
    if config.filter_type == "kalman":
        estimated_states = run_kalman_filter(F, H, Q, R, observations, config.dt)
    elif config.filter_type == "particle":
        estimated_states = run_particle_filter_cv(F, H, Q, R, observations, config)
    else:
        raise ValueError(f"Filter {config.filter_type} not supported for constant velocity")
    
    return {
        "true_states": true_states,
        "estimated_states": estimated_states,
        "observations": observations,
        "times": np.arange(config.n_timesteps) * config.dt,
    }


def run_pendulum_tracking(config: TrackingConfig) -> dict:
    """Run pendulum tracking with EKF or particle filter."""
    
    rng = np.random.default_rng(config.seed)
    
    # Ground truth simulation
    true_states = []
    observations = []
    
    # Initial state: [angle, angular_velocity]
    x_true = np.array([config.initial_angle, config.initial_velocity])
    
    # Dynamics
    f, F_jac = pendulum_dynamics(config.dt, config.gravity, config.pendulum_length)
    h, H_jac = angle_observation_model()
    
    Q = np.eye(2) * config.process_noise**2
    R = np.array([[config.obs_noise**2]])
    
    for t in range(config.n_timesteps):
        # True dynamics with process noise
        w = rng.multivariate_normal(np.zeros(2), Q)
        x_true = f(x_true, None) + w
        true_states.append(x_true.copy())
        
        # Noisy observation
        v = rng.normal(0, config.obs_noise)
        z = h(x_true) + v
        observations.append(z[0])
    
    true_states = np.array(true_states)
    observations = np.array(observations)
    
    # Run filter
    if config.filter_type == "ekf":
        estimated_states = run_ekf(f, h, F_jac, H_jac, Q, R, observations, config)
    elif config.filter_type == "particle":
        estimated_states = run_particle_filter_pendulum(f, h, Q, R, observations, config)
    else:
        raise ValueError(f"Filter {config.filter_type} not supported for pendulum")
    
    return {
        "true_states": true_states,
        "estimated_states": estimated_states,
        "observations": observations,
        "times": np.arange(config.n_timesteps) * config.dt,
    }


def run_kalman_filter(F, H, Q, R, observations, dt):
    """Run Kalman filter on observations."""
    kf = KalmanFilter(F, H, Q, R)
    kf.initialize(x0=np.array([0, 0]), P0=np.eye(2))
    
    estimated_states = []
    for z in observations:
        kf.predict()
        kf.update(np.array([z]))
        x, _ = kf.get_state()
        estimated_states.append(x)
    
    return np.array(estimated_states)


def run_ekf(f, h, F_jac, H_jac, Q, R, observations, config):
    """Run Extended Kalman Filter."""
    ekf = ExtendedKalmanFilter(f, h, F_jac, H_jac, Q, R)
    ekf.initialize(
        x0=np.array([config.initial_angle, config.initial_velocity]),
        P0=np.eye(2)
    )
    
    estimated_states = []
    for z in observations:
        ekf.predict()
        ekf.update(np.array([z]))
        x, _ = ekf.get_state()
        estimated_states.append(x)
    
    return np.array(estimated_states)


def run_particle_filter_cv(F, H, Q, R, observations, config):
    """Run particle filter for constant velocity."""
    
    def f_stochastic(x, u, rng):
        return F @ x + rng.multivariate_normal(np.zeros(2), Q)
    
    def h_func(x):
        return H @ x
    
    likelihood = gaussian_likelihood(R)
    
    pf = ParticleFilter(f_stochastic, h_func, likelihood, config.n_particles)
    pf.initialize(
        mean=np.array([0, 0]),
        cov=np.eye(2),
        rng=np.random.default_rng(config.seed)
    )
    
    estimated_states = []
    for z in observations:
        pf.predict()
        pf.update(np.array([z]))
        pf.resample()
        x, _ = pf.get_state_estimate()
        estimated_states.append(x)
    
    return np.array(estimated_states)


def run_particle_filter_pendulum(f, h, Q, R, observations, config):
    """Run particle filter for pendulum."""
    
    f_stochastic = create_process_noise_wrapper(f, Q)
    likelihood = gaussian_likelihood(R)
    
    pf = ParticleFilter(f_stochastic, h, likelihood, config.n_particles)
    pf.initialize(
        mean=np.array([config.initial_angle, config.initial_velocity]),
        cov=np.eye(2) * 0.1,
        rng=np.random.default_rng(config.seed)
    )
    
    estimated_states = []
    for z in observations:
        pf.predict()
        pf.update(np.array([z]))
        pf.resample()
        x, _ = pf.get_state_estimate()
        estimated_states.append(x)
    
    return np.array(estimated_states)


def save_tracking_plots(results, output_dir, config):
    """Save tracking visualization plots."""
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    times = results["times"]
    true_states = results["true_states"]
    estimated_states = results["estimated_states"]
    observations = results["observations"]
    
    # Plot position/angle
    axes[0].plot(times, true_states[:, 0], 'k-', label='True', linewidth=2)
    axes[0].plot(times, estimated_states[:, 0], 'r--', label='Estimated', linewidth=2)
    axes[0].scatter(times, observations, c='blue', s=10, alpha=0.5, label='Observations')
    axes[0].set_ylabel('Position/Angle')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f'{config.system.replace("_", " ").title()} Tracking ({config.filter_type.upper()})')
    
    # Plot velocity
    axes[1].plot(times, true_states[:, 1], 'k-', label='True', linewidth=2)
    axes[1].plot(times, estimated_states[:, 1], 'r--', label='Estimated', linewidth=2)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Velocity')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'tracking_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Error plot
    fig, ax = plt.subplots(figsize=(12, 4))
    errors = np.abs(true_states - estimated_states)
    ax.plot(times, errors[:, 0], label='Position/Angle Error')
    ax.plot(times, errors[:, 1], label='Velocity Error')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Absolute Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Estimation Errors')
    plt.tight_layout()
    plt.savefig(output_dir / 'tracking_errors.png', dpi=150, bbox_inches='tight')
    plt.close()


@app.command()
def run_forecasting_demo(
    config_path: Path = typer.Option(
        "configs/forecasting_default.yaml",
        help="Path to config file"
    ),
    seed: int = typer.Option(42, help="Random seed"),
):
    """Run time series forecasting with backtesting."""
    
    config = load_config(config_path, ForecastingConfig)
    config.seed = seed
    set_seed(seed)
    
    print(f"=== Forecasting Demo: {config.method} ===")
    print(f"Train size: {config.train_size}, Test size: {config.test_size}")
    
    # Generate synthetic data
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 20, config.n_points)
    y = np.sin(t) + 0.1 * t + rng.normal(0, config.noise_level, config.n_points)
    
    # Setup backtesting
    backtest = RollingWindowBacktest(
        train_size=config.train_size,
        test_size=config.test_size,
        step_size=config.step_size
    )
    
    # Select forecast method
    if config.method == "naive":
        forecast_fn = naive_forecast
    elif config.method == "moving_average":
        forecast_fn = lambda y_train, h: moving_average_forecast(y_train, h, config.ma_window)
    else:
        raise ValueError(f"Unknown method: {config.method}")
    
    # Run backtesting
    results = backtest.run(y, forecast_fn, verbose=True)
    
    print(f"\n=== Results ===")
    print(f"Mean MAE: {results['metrics']['mean_mae']:.4f} ± {results['metrics']['std_mae']:.4f}")
    print(f"Mean RMSE: {results['metrics']['mean_rmse']:.4f} ± {results['metrics']['std_rmse']:.4f}")
    
    # Save plots
    if config.save_plots:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_forecasting_plots(y, results, output_dir, config)
        print(f"\nPlots saved to {output_dir}")
    
    print("\n✓ Forecasting demo complete!")


def save_forecasting_plots(y, results, output_dir, config):
    """Save forecasting visualization plots."""
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot time series with forecasts
    axes[0].plot(y, 'k-', label='Actual', alpha=0.7)
    axes[0].set_ylabel('Value')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f'Time Series Forecasting ({config.method})')
    
    # Plot MAE over windows
    axes[1].plot(results['mae'], 'o-', label='MAE per window')
    axes[1].axhline(results['metrics']['mean_mae'], color='r', linestyle='--', label='Mean MAE')
    axes[1].set_xlabel('Window')
    axes[1].set_ylabel('MAE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'forecasting_results.png', dpi=150, bbox_inches='tight')
    plt.close()


@app.command()
def run_hybrid_demo(
    seed: int = typer.Option(42, help="Random seed"),
    n_train: int = typer.Option(500, help="Number of training samples"),
    n_epochs: int = typer.Option(200, help="Training epochs"),
    complexity: str = typer.Option("moderate", help="Measurement complexity: simple/moderate/complex"),
    output_dir: Path = typer.Option("reports/hybrid", help="Output directory"),
):
    """Run hybrid neural network + state estimation demo.
    
    Demonstrates learning an unknown measurement function with a neural network
    and integrating it with EKF for state estimation.
    """
    import torch
    
    set_seed(seed)
    torch.manual_seed(seed)
    
    print("="*60)
    print("Hybrid Model Demo: Neural Network + EKF")
    print("="*60)
    print(f"Seed: {seed}")
    print(f"Training samples: {n_train}")
    print(f"Measurement complexity: {complexity}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup system
    dt = 0.1
    process_noise = 0.02
    obs_noise = 0.1
    n_steps = 100
    
    F = np.array([[1, dt], [0, 1]])  # Constant velocity dynamics
    Q = np.eye(2) * process_noise**2
    R = np.array([[obs_noise**2]])
    
    def f_dynamics(x, u):
        return F @ x
    
    def F_jacobian(x):
        return F
    
    # Get measurement function (unknown to filter)
    h_true, h_linear, h_jac_true = create_nonlinear_measurement_system(complexity)
    
    print(f"\nSystem: Linear dynamics, nonlinear measurement (complexity={complexity})")
    
    # Generate training data
    print("\n[1/4] Generating training data...")
    rng = np.random.default_rng(seed)
    X_train, Z_train = generate_training_data(
        f_dynamics=None,
        h_true=h_true,
        n_samples=n_train,
        noise_std=obs_noise,
        rng=rng,
        x_range=(-3, 3),
    )
    
    # Train hybrid EKF
    print("\n[2/4] Training measurement network...")
    config = HybridEKFConfig(
        n_states=2,
        n_obs=1,
        hidden_dims=[64, 32],
        learning_rate=0.005,
        n_epochs=n_epochs,
        batch_size=32,
    )
    
    hybrid_ekf = HybridEKF(f_dynamics, F_jacobian, Q, R, config)
    losses = hybrid_ekf.train_measurement_model(X_train, Z_train, verbose=True)
    
    # Generate test trajectory
    print("\n[3/4] Running filter comparison...")
    rng_test = np.random.default_rng(seed + 100)  # Different seed
    
    true_states = []
    observations = []
    x_true = np.array([0.0, 1.0])
    
    for _ in range(n_steps):
        w = rng_test.multivariate_normal(np.zeros(2), Q)
        x_true = F @ x_true + w
        true_states.append(x_true.copy())
        
        v = rng_test.normal(0, obs_noise)
        z = h_true(x_true) + v
        observations.append(z[0])
    
    true_states = np.array(true_states)
    observations = np.array(observations)
    
    # Run filters
    # Hybrid EKF
    hybrid_ekf.initialize(x0=np.array([0, 0]), P0=np.eye(2))
    hybrid_estimates = []
    for z in observations:
        hybrid_ekf.predict()
        hybrid_ekf.update(np.array([z]))
        x, _ = hybrid_ekf.get_state()
        hybrid_estimates.append(x)
    hybrid_estimates = np.array(hybrid_estimates)
    
    # EKF with true Jacobian (oracle)
    ekf_true = ExtendedKalmanFilter(f_dynamics, h_true, F_jacobian, h_jac_true, Q, R)
    ekf_true.initialize(x0=np.array([0, 0]), P0=np.eye(2))
    ekf_true_estimates = []
    for z in observations:
        ekf_true.predict()
        ekf_true.update(np.array([z]))
        x, _ = ekf_true.get_state()
        ekf_true_estimates.append(x)
    ekf_true_estimates = np.array(ekf_true_estimates)
    
    # EKF with wrong (linear) model
    def h_linear_jac(x):
        return np.array([[1, 0]])
    
    ekf_linear = ExtendedKalmanFilter(f_dynamics, h_linear, F_jacobian, h_linear_jac, Q, R)
    ekf_linear.initialize(x0=np.array([0, 0]), P0=np.eye(2))
    ekf_linear_estimates = []
    for z in observations:
        ekf_linear.predict()
        ekf_linear.update(np.array([z]))
        x, _ = ekf_linear.get_state()
        ekf_linear_estimates.append(x)
    ekf_linear_estimates = np.array(ekf_linear_estimates)
    
    # Compute metrics
    def rmse(true, est):
        return np.sqrt(np.mean((true - est)**2))
    
    rmse_hybrid = rmse(true_states, hybrid_estimates)
    rmse_true = rmse(true_states, ekf_true_estimates)
    rmse_linear = rmse(true_states, ekf_linear_estimates)
    
    print("\n[4/4] Results")
    print("="*60)
    print(f"{'Method':<30} {'RMSE':>12}")
    print("-"*60)
    print(f"{'Hybrid EKF (learned h)':<30} {rmse_hybrid:>12.4f}")
    print(f"{'EKF (true Jacobian - oracle)':<30} {rmse_true:>12.4f}")
    print(f"{'EKF (linear h - wrong model)':<30} {rmse_linear:>12.4f}")
    print("="*60)
    
    improvement = (rmse_linear - rmse_hybrid) / rmse_linear * 100
    print(f"\nHybrid improves over wrong-model EKF by {improvement:.1f}%")
    
    # Save plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    times = np.arange(n_steps) * dt
    
    # Position
    ax = axes[0, 0]
    ax.plot(times, true_states[:, 0], 'k-', label='True', linewidth=2)
    ax.plot(times, hybrid_estimates[:, 0], 'r--', label='Hybrid EKF', linewidth=2)
    ax.plot(times, ekf_true_estimates[:, 0], 'g:', label='EKF (true)', linewidth=2)
    ax.plot(times, ekf_linear_estimates[:, 0], 'b-.', label='EKF (linear)', alpha=0.7)
    ax.set_ylabel('Position')
    ax.legend()
    ax.set_title('Position Tracking')
    ax.grid(True, alpha=0.3)
    
    # Velocity
    ax = axes[0, 1]
    ax.plot(times, true_states[:, 1], 'k-', label='True', linewidth=2)
    ax.plot(times, hybrid_estimates[:, 1], 'r--', label='Hybrid EKF', linewidth=2)
    ax.plot(times, ekf_true_estimates[:, 1], 'g:', label='EKF (true)', linewidth=2)
    ax.set_ylabel('Velocity')
    ax.legend()
    ax.set_title('Velocity Tracking')
    ax.grid(True, alpha=0.3)
    
    # Training loss
    ax = axes[1, 0]
    ax.semilogy(losses)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Measurement Network Training')
    ax.grid(True, alpha=0.3)
    
    # RMSE bar chart
    ax = axes[1, 1]
    methods = ['Hybrid\nEKF', 'EKF\n(true h)', 'EKF\n(linear h)']
    rmses = [rmse_hybrid, rmse_true, rmse_linear]
    colors = ['red', 'green', 'blue']
    bars = ax.bar(methods, rmses, color=colors, alpha=0.7)
    ax.set_ylabel('RMSE')
    ax.set_title('Overall RMSE Comparison')
    for bar, r in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{r:.3f}', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'hybrid_demo_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save metrics
    import json
    metrics = {
        "seed": seed,
        "n_train": n_train,
        "n_epochs": n_epochs,
        "complexity": complexity,
        "rmse_hybrid": float(rmse_hybrid),
        "rmse_ekf_true": float(rmse_true),
        "rmse_ekf_linear": float(rmse_linear),
        "improvement_pct": float(improvement),
        "final_training_loss": float(losses[-1]),
    }
    with open(output_dir / "hybrid_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ Results saved to {output_dir}")
    print("  - hybrid_demo_results.png")
    print("  - hybrid_metrics.json")


if __name__ == "__main__":
    app()

"""CLI entry point for Physics-Informed ML experiments."""

import typer
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from modules._import_helper import safe_import_from

set_seed = safe_import_from('00_repo_standards.src.mlphys_core', 'set_seed')

# ODE PINN
(HarmonicOscillatorConfig, HarmonicOscillatorPINN,
 solve_harmonic_oscillator_scipy, analytical_harmonic_oscillator,
 compute_energy) = safe_import_from(
    '07_physics_informed_ml.src.ode_pinn',
    'HarmonicOscillatorConfig', 'HarmonicOscillatorPINN',
    'solve_harmonic_oscillator_scipy', 'analytical_harmonic_oscillator',
    'compute_energy'
)

# PDE PINN
(HeatEquationConfig, HeatEquationPINN) = safe_import_from(
    '07_physics_informed_ml.src.pde_pinn',
    'HeatEquationConfig', 'HeatEquationPINN'
)

# Constrained learning
(ConservationConfig, ConservationConstrainedNN,
 generate_pendulum_data, compute_energy_violation) = safe_import_from(
    '07_physics_informed_ml.src.constrained_learning',
    'ConservationConfig', 'ConservationConstrainedNN',
    'generate_pendulum_data', 'compute_energy_violation'
)


app = typer.Typer(help="Physics-Informed ML experiments")


@app.command()
def run_pinn_ode(
    omega: float = typer.Option(1.0, help="Angular frequency"),
    epochs: int = typer.Option(5000, help="Training epochs"),
    seed: int = typer.Option(42, help="Random seed"),
    output_dir: Path = typer.Option("modules/07_physics_informed_ml/reports", help="Output directory"),
):
    """
    Train PINN on harmonic oscillator ODE and compare to scipy.
    
    Saves: convergence plot, solution comparison, error analysis
    """
    print(f"\n{'='*60}")
    print(f"ODE PINN: Harmonic Oscillator (ω={omega})")
    print(f"{'='*60}\n")
    
    set_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure PINN
    config = HarmonicOscillatorConfig(
        omega=omega,
        epochs=epochs,
        n_collocation=200,
    )
    
    # Train PINN
    print("Training PINN...")
    pinn = HarmonicOscillatorPINN(config)
    history = pinn.train(verbose=500)
    
    # Baseline: scipy solver
    print("\nRunning scipy baseline...")
    t_eval = np.linspace(0, config.t_max, 200)
    x_scipy, v_scipy = solve_harmonic_oscillator_scipy(
        omega=omega,
        x0=config.x0,
        v0=config.v0,
        t_eval=t_eval,
    )
    
    # Analytical solution
    x_analytical, v_analytical = analytical_harmonic_oscillator(
        omega=omega,
        x0=config.x0,
        v0=config.v0,
        t=t_eval,
    )
    
    # PINN predictions
    x_pinn, v_pinn = pinn.predict_with_velocity(t_eval)
    
    # Compute errors
    l2_error_scipy = np.sqrt(np.mean((x_pinn - x_scipy) ** 2))
    l2_error_analytical = np.sqrt(np.mean((x_pinn - x_analytical) ** 2))
    
    # Energy conservation
    E_pinn = compute_energy(x_pinn, v_pinn, omega)
    E_analytical = compute_energy(x_analytical, v_analytical, omega)
    energy_violation = np.std(E_pinn) / np.mean(E_analytical) * 100
    
    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"L2 Error vs scipy:      {l2_error_scipy:.6f}")
    print(f"L2 Error vs analytical: {l2_error_analytical:.6f}")
    print(f"Energy violation:       {energy_violation:.2f}%")
    print(f"Final loss:             {history['loss'][-1]:.6f}")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Training convergence
    axes[0, 0].semilogy(history['loss'], label='Total')
    axes[0, 0].semilogy(history['loss_physics'], label='Physics')
    axes[0, 0].semilogy(history['loss_ic'], label='IC')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Convergence')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Solution comparison
    axes[0, 1].plot(t_eval, x_analytical, 'k-', label='Analytical', linewidth=2)
    axes[0, 1].plot(t_eval, x_scipy, 'b--', label='scipy', linewidth=1.5)
    axes[0, 1].plot(t_eval, x_pinn, 'r:', label='PINN', linewidth=2)
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Position')
    axes[0, 1].set_title('Solution Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Error vs time
    error_scipy = np.abs(x_pinn - x_scipy)
    error_analytical = np.abs(x_pinn - x_analytical)
    axes[1, 0].semilogy(t_eval, error_scipy, label='vs scipy')
    axes[1, 0].semilogy(t_eval, error_analytical, label='vs analytical')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Absolute Error')
    axes[1, 0].set_title('Pointwise Error')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Phase space
    axes[1, 1].plot(x_analytical, v_analytical, 'k-', label='Analytical', linewidth=2)
    axes[1, 1].plot(x_pinn, v_pinn, 'r:', label='PINN', linewidth=2)
    axes[1, 1].set_xlabel('Position')
    axes[1, 1].set_ylabel('Velocity')
    axes[1, 1].set_title('Phase Space')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    axes[1, 1].axis('equal')
    
    plt.tight_layout()
    plot_path = output_dir / f"ode_pinn_omega{omega}_seed{seed}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    
    # Save metrics
    metrics = {
        'l2_error_scipy': float(l2_error_scipy),
        'l2_error_analytical': float(l2_error_analytical),
        'energy_violation_percent': float(energy_violation),
        'final_loss': float(history['loss'][-1]),
    }
    
    metrics_path = output_dir / f"ode_pinn_omega{omega}_seed{seed}_metrics.txt"
    with open(metrics_path, 'w') as f:
        for key, val in metrics.items():
            f.write(f"{key}: {val}\n")
    
    print(f"Metrics saved to: {metrics_path}\n")


@app.command()
def run_pinn_pde(
    alpha: float = typer.Option(0.01, help="Thermal diffusivity"),
    initial_condition: str = typer.Option("gaussian", help="Initial condition type"),
    epochs: int = typer.Option(10000, help="Training epochs"),
    seed: int = typer.Option(42, help="Random seed"),
    output_dir: Path = typer.Option("modules/07_physics_informed_ml/reports", help="Output directory"),
):
    """
    Train PINN on 1D heat equation PDE.
    
    Saves: convergence plot, solution heatmap, error analysis
    """
    print(f"\n{'='*60}")
    print(f"PDE PINN: 1D Heat Equation (α={alpha})")
    print(f"{'='*60}\n")
    
    set_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure PINN
    config = HeatEquationConfig(
        alpha=alpha,
        initial_condition=initial_condition,
        epochs=epochs,
    )
    
    # Train PINN
    print("Training PINN...")
    pinn = HeatEquationPINN(config)
    history = pinn.train(verbose=1000)
    
    # Create evaluation grid
    x_eval = np.linspace(config.x_min, config.x_max, 100)
    t_eval = np.linspace(0, config.t_max, 100)
    xx, tt = np.meshgrid(x_eval, t_eval)
    
    # PINN predictions
    u_pinn = pinn.predict(xx.flatten(), tt.flatten()).reshape(xx.shape)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Final loss:         {history['loss'][-1]:.6f}")
    print(f"Final physics loss: {history['loss_physics'][-1]:.6f}")
    print(f"Final BC loss:      {history['loss_bc'][-1]:.6f}")
    print(f"Final IC loss:      {history['loss_ic'][-1]:.6f}")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Training convergence
    axes[0, 0].semilogy(history['loss'], label='Total')
    axes[0, 0].semilogy(history['loss_physics'], label='Physics')
    axes[0, 0].semilogy(history['loss_bc'], label='BC')
    axes[0, 0].semilogy(history['loss_ic'], label='IC')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Convergence')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Solution heatmap
    im = axes[0, 1].contourf(xx, tt, u_pinn, levels=50, cmap='hot')
    axes[0, 1].set_xlabel('Space (x)')
    axes[0, 1].set_ylabel('Time (t)')
    axes[0, 1].set_title('PINN Solution u(x, t)')
    plt.colorbar(im, ax=axes[0, 1])
    
    # Spatial profiles at different times
    for t_idx, t_val in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
        t_idx_grid = int(t_val * (len(t_eval) - 1))
        axes[1, 0].plot(x_eval, u_pinn[t_idx_grid, :], label=f't={t_val:.2f}')
    axes[1, 0].set_xlabel('Space (x)')
    axes[1, 0].set_ylabel('Temperature u')
    axes[1, 0].set_title('Spatial Profiles')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Temporal profiles at different locations
    for x_idx, x_val in enumerate([0.25, 0.5, 0.75]):
        x_idx_grid = int(x_val * (len(x_eval) - 1))
        axes[1, 1].plot(t_eval, u_pinn[:, x_idx_grid], label=f'x={x_val:.2f}')
    axes[1, 1].set_xlabel('Time (t)')
    axes[1, 1].set_ylabel('Temperature u')
    axes[1, 1].set_title('Temporal Profiles')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plot_path = output_dir / f"pde_pinn_alpha{alpha}_{initial_condition}_seed{seed}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    
    # Save metrics
    metrics = {
        'final_loss': float(history['loss'][-1]),
        'final_physics_loss': float(history['loss_physics'][-1]),
        'final_bc_loss': float(history['loss_bc'][-1]),
        'final_ic_loss': float(history['loss_ic'][-1]),
    }
    
    metrics_path = output_dir / f"pde_pinn_alpha{alpha}_{initial_condition}_seed{seed}_metrics.txt"
    with open(metrics_path, 'w') as f:
        for key, val in metrics.items():
            f.write(f"{key}: {val}\n")
    
    print(f"Metrics saved to: {metrics_path}\n")


if __name__ == "__main__":
    app()

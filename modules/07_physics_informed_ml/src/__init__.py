"""Physics-Informed Machine Learning module."""

from .pinn_base import (
    PINN,
    PINNConfig,
    compute_gradient,
    pinn_loss,
    train_pinn,
)

from .ode_pinn import (
    HarmonicOscillatorConfig,
    HarmonicOscillatorPINN,
    solve_harmonic_oscillator_scipy,
    analytical_harmonic_oscillator,
    compute_energy,
)

from .pde_pinn import (
    HeatEquationConfig,
    HeatEquationPINN,
    solve_heat_equation_finite_difference,
)

from .constrained_learning import (
    ConservationConfig,
    ConservationConstrainedNN,
    generate_pendulum_data,
    compute_energy_violation,
)

__all__ = [
    # Base
    "PINN",
    "PINNConfig",
    "compute_gradient",
    "pinn_loss",
    "train_pinn",
    # ODE
    "HarmonicOscillatorConfig",
    "HarmonicOscillatorPINN",
    "solve_harmonic_oscillator_scipy",
    "analytical_harmonic_oscillator",
    "compute_energy",
    # PDE
    "HeatEquationConfig",
    "HeatEquationPINN",
    "solve_heat_equation_finite_difference",
    # Constrained learning
    "ConservationConfig",
    "ConservationConstrainedNN",
    "generate_pendulum_data",
    "compute_energy_violation",
]

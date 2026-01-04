"""Configuration dataclasses for Module 05."""

from dataclasses import dataclass
from pathlib import Path
from modules._import_helper import safe_import_from

ExperimentConfig = safe_import_from(
    '00_repo_standards.src.mlphys_core.config',
    'ExperimentConfig'
)


@dataclass
class MCIntegrationConfig(ExperimentConfig):
    """Configuration for Monte Carlo integration experiments."""
    
    # MC parameters
    n_samples: int = 10000
    confidence_level: float = 0.95
    
    # Convergence analysis
    min_samples: int = 100
    max_samples: int = 50000
    n_trials: int = 50
    
    # Integration bounds
    lower_bound: float = 0.0
    upper_bound: float = 1.0


@dataclass
class VarianceReductionConfig(ExperimentConfig):
    """Configuration for variance reduction experiments."""
    
    # Sample sizes for comparison
    n_samples: int = 10000
    sample_sizes_comparison: list[int] = None
    
    # Methods to compare
    use_importance_sampling: bool = True
    use_control_variates: bool = True
    use_antithetic: bool = True
    
    # Importance sampling
    proposal_shift: float = 0.0  # Mean shift for Gaussian proposal
    proposal_scale: float = 1.0  # Scale for Gaussian proposal
    
    # Control variates
    control_mean: float = 0.0  # Known mean of control variable
    
    def __post_init__(self):
        super().__post_init__()
        if self.sample_sizes_comparison is None:
            self.sample_sizes_comparison = [100, 500, 1000, 2000, 5000, 10000]


@dataclass
class RareEventConfig(ExperimentConfig):
    """Configuration for rare event estimation experiments."""
    
    # Event parameters
    distribution: str = 'normal'  # 'normal' or 'exponential'
    threshold: float = 4.0  # Event occurs when X > threshold
    
    # MC parameters
    n_samples: int = 50000
    n_trials: int = 100
    
    # Methods
    method: str = 'importance_sampling'  # 'naive' or 'importance_sampling'
    
    # Adaptive sampling
    use_adaptive: bool = False
    n_pilot: int = 1000
    n_main: int = 9000
    target_relative_error: float = 0.05
    max_iterations: int = 10
    
    # Thresholds for comparison
    thresholds_comparison: list[float] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.thresholds_comparison is None:
            self.thresholds_comparison = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]


@dataclass
class SyntheticDataConfig(ExperimentConfig):
    """Configuration for synthetic data generation."""
    
    # Dataset parameters
    n_samples: int = 1000
    noise_level: float = 0.1
    
    # Physics datasets
    dataset_type: str = 'oscillator'  # 'oscillator', 'projectile', 'heat', 'pendulum'
    
    # Oscillator parameters
    mass: float = 1.0
    spring_constant: float = 10.0
    damping: float = 0.5
    initial_amplitude: float = 1.0
    
    # Projectile parameters
    gravity: float = 9.81
    drag_coefficient: float = 0.1
    
    # Heat diffusion parameters
    thermal_diffusivity: float = 0.1
    length: float = 1.0
    
    # Pendulum parameters
    pendulum_length: float = 1.0
    pendulum_mass: float = 1.0
    
    # Stochastic processes
    process_type: str = 'brownian'  # 'brownian', 'ou', 'levy'
    n_steps: int = 1000
    dt: float = 0.01
    
    # Brownian motion
    drift: float = 0.0
    diffusion: float = 1.0
    
    # OU process
    theta: float = 1.0  # Mean reversion rate
    ou_mean: float = 0.0  # Long-term mean
    
    # Lévy flight
    levy_alpha: float = 1.5  # Stability parameter (1 < alpha <= 2)
    
    # Correlated noise
    correlation_length: float = 5.0

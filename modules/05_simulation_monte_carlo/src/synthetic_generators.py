"""Physics-inspired synthetic data generators."""

import numpy as np
from typing import Optional, Tuple, Callable
from dataclasses import dataclass


@dataclass
class PhysicsDataset:
    """Synthetic dataset with ground truth."""
    features: np.ndarray
    targets: np.ndarray
    ground_truth_func: Callable
    noise_level: float
    description: str


def generate_brownian_motion(
    n_steps: int = 1000,
    dt: float = 0.01,
    mu: float = 0.0,
    sigma: float = 1.0,
    x0: float = 0.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Brownian motion (Wiener process with drift).
    
    dX_t = μ dt + σ dW_t
    
    Args:
        n_steps: Number of time steps
        dt: Time step size
        mu: Drift coefficient
        sigma: Diffusion coefficient
        x0: Initial value
        seed: Random seed
        
    Returns:
        (times, positions)
        
    Example:
        >>> t, x = generate_brownian_motion(n_steps=1000, dt=0.01, mu=0.1, sigma=0.5)
        >>> # x follows Brownian motion with drift
    """
    rng = np.random.default_rng(seed)
    
    # Generate increments
    dW = rng.normal(0, np.sqrt(dt), n_steps)
    increments = mu * dt + sigma * dW
    
    # Cumulative sum to get path
    x = np.zeros(n_steps + 1)
    x[0] = x0
    x[1:] = x0 + np.cumsum(increments)
    
    t = np.arange(n_steps + 1) * dt
    
    return t, x


def generate_ou_process(
    n_steps: int = 1000,
    dt: float = 0.01,
    theta: float = 1.0,
    mu: float = 0.0,
    sigma: float = 0.5,
    x0: Optional[float] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Ornstein-Uhlenbeck process (mean-reverting).
    
    dX_t = θ(μ - X_t) dt + σ dW_t
    
    Models: interest rates, velocity, etc.
    
    Args:
        n_steps: Number of steps
        dt: Time step
        theta: Mean reversion rate
        mu: Long-term mean
        sigma: Volatility
        x0: Initial value (if None, use mu)
        seed: Random seed
        
    Returns:
        (times, positions)
    """
    rng = np.random.default_rng(seed)
    
    if x0 is None:
        x0 = mu
    
    x = np.zeros(n_steps + 1)
    x[0] = x0
    
    for i in range(n_steps):
        dW = rng.normal(0, np.sqrt(dt))
        x[i+1] = x[i] + theta * (mu - x[i]) * dt + sigma * dW
    
    t = np.arange(n_steps + 1) * dt
    
    return t, x


def generate_levy_flight(
    n_steps: int = 1000,
    alpha: float = 1.5,
    scale: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Lévy flight (heavy-tailed random walk).
    
    Step sizes follow stable distribution with index α ∈ (0, 2].
    α = 2: Gaussian (Brownian), α < 2: heavy tails.
    
    Models: animal foraging, turbulence, financial jumps.
    
    Args:
        n_steps: Number of steps
        alpha: Stability index (0 < α ≤ 2)
        scale: Scale parameter
        seed: Random seed
        
    Returns:
        (step_indices, positions)
    """
    rng = np.random.default_rng(seed)
    
    # Generate Lévy-distributed step sizes
    # Approximation: mixture of Gaussian + rare large jumps
    steps = np.zeros(n_steps)
    
    for i in range(n_steps):
        if rng.random() < 0.05:  # 5% large jumps
            # Heavy tail: power-law
            u = rng.uniform(0, 1)
            step = scale * np.sign(rng.normal()) * (u ** (-1/alpha))
        else:
            # Normal small steps
            step = rng.normal(0, scale)
        steps[i] = step
    
    # Cumulative path
    positions = np.cumsum(steps)
    positions = np.concatenate([[0], positions])
    
    return np.arange(n_steps + 1), positions


class PhysicsDataGenerator:
    """
    Generator for physics-inspired regression/classification datasets.
    
    Provides synthetic data with known ground truth for ML benchmarks.
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
    
    def damped_harmonic_oscillator(
        self,
        n_samples: int = 1000,
        noise_level: float = 0.1,
    ) -> PhysicsDataset:
        """
        Damped harmonic oscillator: x''(t) + 2ζω₀x'(t) + ω₀²x(t) = 0.
        
        Solution: x(t) = A e^(-ζω₀t) cos(ω_d t + φ)
        
        Task: Given (t, x'(t)), predict x(t).
        
        Args:
            n_samples: Number of samples
            noise_level: Observation noise std
            
        Returns:
            PhysicsDataset
        """
        # Parameters
        zeta = 0.1  # Damping ratio
        omega_0 = 2.0  # Natural frequency
        omega_d = omega_0 * np.sqrt(1 - zeta**2)  # Damped frequency
        
        A = 1.0
        phi = 0.0
        
        # Time points
        t = self.rng.uniform(0, 10, n_samples)
        
        # Ground truth
        def position(t):
            return A * np.exp(-zeta * omega_0 * t) * np.cos(omega_d * t + phi)
        
        def velocity(t):
            return -A * omega_0 * (
                zeta * np.exp(-zeta * omega_0 * t) * np.cos(omega_d * t + phi) +
                np.sqrt(1 - zeta**2) * np.exp(-zeta * omega_0 * t) * np.sin(omega_d * t + phi)
            )
        
        # Features: (time, velocity)
        v_true = velocity(t)
        v_noisy = v_true + self.rng.normal(0, noise_level, n_samples)
        features = np.column_stack([t, v_noisy])
        
        # Targets: position
        x_true = position(t)
        targets = x_true + self.rng.normal(0, noise_level, n_samples)
        
        return PhysicsDataset(
            features=features,
            targets=targets,
            ground_truth_func=position,
            noise_level=noise_level,
            description="Damped harmonic oscillator: predict position from time and velocity",
        )
    
    def projectile_motion(
        self,
        n_samples: int = 500,
        noise_level: float = 0.5,
    ) -> PhysicsDataset:
        """
        Projectile motion with air resistance.
        
        Task: Given (v₀, angle), predict range.
        
        Args:
            n_samples: Number of trajectories
            noise_level: Measurement noise
            
        Returns:
            PhysicsDataset
        """
        g = 9.8  # m/s²
        k = 0.1  # Air resistance coefficient
        
        # Random initial conditions
        v0 = self.rng.uniform(10, 30, n_samples)  # m/s
        theta = self.rng.uniform(20, 70, n_samples) * np.pi / 180  # radians
        
        # Ground truth range (numerical integration would be exact, use approximation)
        def range_func(v0, theta):
            # Without air resistance: v0^2 sin(2θ) / g
            # With resistance: reduced by factor
            ideal_range = v0**2 * np.sin(2*theta) / g
            reduction = np.exp(-k * ideal_range / v0)
            return ideal_range * reduction
        
        true_range = range_func(v0, theta)
        
        # Features: (v0, angle)
        features = np.column_stack([v0, theta])
        
        # Targets: range with noise
        targets = true_range + self.rng.normal(0, noise_level, n_samples)
        
        return PhysicsDataset(
            features=features,
            targets=targets,
            ground_truth_func=lambda x: range_func(x[:, 0], x[:, 1]),
            noise_level=noise_level,
            description="Projectile motion: predict range from initial velocity and angle",
        )
    
    def heat_diffusion_1d(
        self,
        n_samples: int = 1000,
        noise_level: float = 0.05,
    ) -> PhysicsDataset:
        """
        1D heat equation: u_t = α u_xx.
        
        Solution: u(x,t) = exp(-α k² t) sin(kx)
        
        Task: Given (x, t), predict temperature u.
        
        Args:
            n_samples: Number of samples
            noise_level: Noise level
            
        Returns:
            PhysicsDataset
        """
        alpha = 0.1  # Thermal diffusivity
        k = 1.0  # Wave number
        
        # Sample space-time points
        x = self.rng.uniform(0, 2*np.pi, n_samples)
        t = self.rng.uniform(0, 5, n_samples)
        
        # Ground truth
        def temperature(x, t):
            return np.exp(-alpha * k**2 * t) * np.sin(k * x)
        
        u_true = temperature(x, t)
        
        # Features: (x, t)
        features = np.column_stack([x, t])
        
        # Targets: temperature
        targets = u_true + self.rng.normal(0, noise_level, n_samples)
        
        return PhysicsDataset(
            features=features,
            targets=targets,
            ground_truth_func=lambda xt: temperature(xt[:, 0], xt[:, 1]),
            noise_level=noise_level,
            description="1D heat diffusion: predict temperature from position and time",
        )
    
    def pendulum_energy(
        self,
        n_samples: int = 800,
        noise_level: float = 0.1,
    ) -> PhysicsDataset:
        """
        Simple pendulum: predict energy from angle and angular velocity.
        
        E = (1/2) m L² θ'² + m g L (1 - cos(θ))
        
        Args:
            n_samples: Number of samples
            noise_level: Noise level
            
        Returns:
            PhysicsDataset
        """
        m = 1.0  # Mass (kg)
        L = 1.0  # Length (m)
        g = 9.8  # Gravity (m/s²)
        
        # Sample phase space
        theta = self.rng.uniform(-np.pi, np.pi, n_samples)
        theta_dot = self.rng.uniform(-5, 5, n_samples)
        
        # Ground truth energy
        def energy(theta, theta_dot):
            kinetic = 0.5 * m * L**2 * theta_dot**2
            potential = m * g * L * (1 - np.cos(theta))
            return kinetic + potential
        
        E_true = energy(theta, theta_dot)
        
        # Features: (theta, theta_dot)
        features = np.column_stack([theta, theta_dot])
        
        # Targets: energy
        targets = E_true + self.rng.normal(0, noise_level, n_samples)
        
        return PhysicsDataset(
            features=features,
            targets=targets,
            ground_truth_func=lambda x: energy(x[:, 0], x[:, 1]),
            noise_level=noise_level,
            description="Pendulum: predict total energy from angle and angular velocity",
        )


def generate_correlated_noise(
    n_samples: int,
    correlation_length: float = 5.0,
    sigma: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate temporally correlated noise (Gaussian process).
    
    Uses exponential covariance: K(t, t') = σ² exp(-|t-t'|/ℓ).
    
    Args:
        n_samples: Number of samples
        correlation_length: Correlation length ℓ
        sigma: Marginal std
        seed: Random seed
        
    Returns:
        Correlated noise array
    """
    rng = np.random.default_rng(seed)
    
    # Time indices
    t = np.arange(n_samples)
    
    # Covariance matrix
    T1, T2 = np.meshgrid(t, t)
    K = sigma**2 * np.exp(-np.abs(T1 - T2) / correlation_length)
    
    # Add small diagonal for numerical stability
    K += 1e-6 * np.eye(n_samples)
    
    # Cholesky decomposition
    L = np.linalg.cholesky(K)
    
    # Sample from multivariate Gaussian
    z = rng.standard_normal(n_samples)
    noise = L @ z
    
    return noise

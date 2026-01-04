"""Bootstrap Particle Filter for nonlinear/non-Gaussian systems.

Sequential Monte Carlo method that represents the posterior distribution
with a set of weighted particles (samples).

References:
    Arulampalam et al. (2002). "A Tutorial on Particle Filters"
"""

import numpy as np
from typing import Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class ParticleFilter:
    """
    Bootstrap Particle Filter (Sequential Importance Resampling).
    
    Represents posterior p(x_t | y_{1:t}) with N weighted particles.
    
    Algorithm:
        1. Prediction: propagate particles through dynamics
        2. Update: reweight by observation likelihood
        3. Resample: draw new particles according to weights
    
    Args:
        f: State transition function (x, u, rng) -> x_next (stochastic)
        h: Observation function x -> z_expected
        likelihood: Likelihood function (z_observed, z_expected) -> probability
        n_particles: Number of particles
        
    Attributes:
        particles: Current particle set (n_particles, n_states)
        weights: Particle weights (n_particles,)
        
    Example:
        >>> # Nonlinear system with process noise
        >>> def f(x, u, rng):
        ...     # Pendulum with noise
        ...     theta, omega = x
        ...     dt, g, L = 0.1, 9.81, 1.0
        ...     omega_next = omega - (g/L)*np.sin(theta)*dt + rng.normal(0, 0.1)
        ...     theta_next = theta + omega_next*dt
        ...     return np.array([theta_next, omega_next])
        >>> 
        >>> def h(x):
        ...     return np.array([x[0]])  # observe angle
        >>> 
        >>> def likelihood(z_obs, z_pred):
        ...     # Gaussian likelihood
        ...     R = 0.1
        ...     diff = z_obs - z_pred
        ...     return np.exp(-0.5 * diff**2 / R**2) / np.sqrt(2*np.pi*R**2)
        >>> 
        >>> pf = ParticleFilter(f, h, likelihood, n_particles=1000)
        >>> pf.initialize(mean=np.array([0, 0]), cov=np.eye(2), rng=np.random.default_rng(42))
    """
    
    f: Callable[[np.ndarray, Optional[np.ndarray], np.random.Generator], np.ndarray]
    h: Callable[[np.ndarray], np.ndarray]
    likelihood: Callable[[np.ndarray, np.ndarray], float]
    n_particles: int
    resample_threshold: float = 0.5  # N_eff / N threshold for resampling
    
    def __post_init__(self):
        """Initialize particle containers."""
        self.particles: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None
        self.n_states: Optional[int] = None
        self.rng: Optional[np.random.Generator] = None
        
    def initialize(
        self,
        mean: np.ndarray,
        cov: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        """
        Initialize particle filter by sampling from Gaussian.
        
        Args:
            mean: Initial state mean (n_states,)
            cov: Initial state covariance (n_states, n_states)
            rng: NumPy random generator for reproducibility
        """
        self.n_states = mean.shape[0]
        self.rng = rng
        
        # Sample particles from initial distribution
        self.particles = rng.multivariate_normal(
            mean, cov, size=self.n_particles
        )
        
        # Uniform weights initially
        self.weights = np.ones(self.n_particles) / self.n_particles
        
    def predict(self, u: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Prediction step: propagate particles through dynamics.
        
        Each particle evolves independently:
            x_i^{t|t-1} = f(x_i^{t-1|t-1}, u_t, w_i)
        
        Args:
            u: Control input, optional
            
        Returns:
            particles: Predicted particles (n_particles, n_states)
        """
        if self.particles is None or self.rng is None:
            raise ValueError("Filter not initialized. Call initialize() first.")
        
        # Propagate each particle
        for i in range(self.n_particles):
            self.particles[i] = self.f(self.particles[i], u, self.rng)
            
        return self.particles
    
    def update(self, z: np.ndarray) -> np.ndarray:
        """
        Update step: reweight particles based on observation likelihood.
        
        Weight update:
            w_i^t ∝ w_i^{t-1} * p(y_t | x_i^{t|t-1})
        
        Args:
            z: Observation (n_obs,)
            
        Returns:
            weights: Updated weights (n_particles,)
        """
        if self.particles is None:
            raise ValueError("Filter not initialized.")
        
        # Compute likelihood for each particle
        for i in range(self.n_particles):
            z_pred = self.h(self.particles[i])
            self.weights[i] *= self.likelihood(z, z_pred)
            
        # Normalize weights
        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights /= weight_sum
        else:
            # Degeneracy: reset to uniform
            self.weights = np.ones(self.n_particles) / self.n_particles
            
        return self.weights
    
    def resample(self) -> None:
        """
        Resample particles if effective sample size is too low.
        
        Effective sample size:
            N_eff = 1 / sum(w_i^2)
        
        Systematic resampling: low-variance method.
        """
        if self.particles is None or self.weights is None or self.rng is None:
            raise ValueError("Filter not initialized.")
        
        # Check if resampling needed
        n_eff = 1.0 / np.sum(self.weights**2)
        if n_eff > self.resample_threshold * self.n_particles:
            return  # No resampling needed
        
        # Systematic resampling
        cumsum = np.cumsum(self.weights)
        u0 = self.rng.uniform(0, 1.0 / self.n_particles)
        u = u0 + np.arange(self.n_particles) / self.n_particles
        
        indices = np.searchsorted(cumsum, u)
        self.particles = self.particles[indices].copy()
        
        # Reset weights to uniform after resampling
        self.weights = np.ones(self.n_particles) / self.n_particles
        
    def get_state_estimate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get weighted mean and covariance of particles.
        
        Returns:
            mean: Weighted mean state (n_states,)
            cov: Weighted covariance (n_states, n_states)
        """
        if self.particles is None or self.weights is None:
            raise ValueError("Filter not initialized.")
        
        # Weighted mean
        mean = np.sum(self.particles * self.weights[:, np.newaxis], axis=0)
        
        # Weighted covariance
        diff = self.particles - mean
        cov = (self.weights[:, np.newaxis, np.newaxis] * diff[:, :, np.newaxis] * diff[:, np.newaxis, :]).sum(axis=0)
        
        return mean, cov


def gaussian_likelihood(R: np.ndarray) -> Callable:
    """
    Create Gaussian likelihood function for observations.
    
    Args:
        R: Observation noise covariance (n_obs, n_obs)
        
    Returns:
        likelihood: Function (z_obs, z_pred) -> probability
    """
    n_obs = R.shape[0]
    R_inv = np.linalg.inv(R)
    det_R = np.linalg.det(R)
    norm_const = 1.0 / np.sqrt((2 * np.pi)**n_obs * det_R)
    
    def likelihood(z_obs: np.ndarray, z_pred: np.ndarray) -> float:
        """Compute Gaussian likelihood."""
        diff = z_obs - z_pred
        exponent = -0.5 * diff.T @ R_inv @ diff
        return norm_const * np.exp(exponent)
    
    return likelihood


def create_process_noise_wrapper(
    f_deterministic: Callable,
    Q: np.ndarray
) -> Callable:
    """
    Wrap a deterministic dynamics function to add process noise.
    
    Args:
        f_deterministic: Deterministic dynamics (x, u) -> x_next
        Q: Process noise covariance (n_states, n_states)
        
    Returns:
        f_stochastic: Stochastic dynamics (x, u, rng) -> x_next
    """
    def f_stochastic(
        x: np.ndarray,
        u: Optional[np.ndarray],
        rng: np.random.Generator
    ) -> np.ndarray:
        """Add process noise to deterministic dynamics."""
        x_next = f_deterministic(x, u)
        noise = rng.multivariate_normal(np.zeros(x.shape[0]), Q)
        return x_next + noise
    
    return f_stochastic

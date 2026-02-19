"""Hybrid state estimation: neural networks + classical filters.

Combines learned components (neural nets) with model-based filtering (EKF/PF).
Useful when parts of the dynamics or measurement model are unknown/complex.

Key idea: Let NN learn what physics doesn't know, keep what physics does know.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class MeasurementNetwork(nn.Module):
    """
    Neural network that learns a nonlinear measurement function.
    
    Learns: h(x) -> z, where the true measurement is complex/unknown.
    
    Architecture: Simple MLP with configurable layers.
    
    Example:
        >>> # Learn measurement function from state to sensor reading
        >>> net = MeasurementNetwork(n_states=2, n_obs=1, hidden_dims=[32, 16])
        >>> x = torch.randn(10, 2)  # batch of states
        >>> z_pred = net(x)         # predicted measurements
    """
    
    def __init__(
        self,
        n_states: int,
        n_obs: int,
        hidden_dims: List[int] = None,
        activation: str = "tanh",
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 16]
        
        self.n_states = n_states
        self.n_obs = n_obs
        
        # Build layers
        layers = []
        in_dim = n_states
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "relu":
                layers.append(nn.ReLU())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, n_obs))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: state -> measurement."""
        return self.network(x)
    
    def predict_numpy(self, x: np.ndarray) -> np.ndarray:
        """Predict from numpy array (for filter integration)."""
        with torch.no_grad():
            x_t = torch.from_numpy(x.astype(np.float32))
            if x_t.dim() == 1:
                x_t = x_t.unsqueeze(0)
            z = self.network(x_t)
            return z.squeeze(0).numpy()
    
    def jacobian_numpy(self, x: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian ∂h/∂x at point x (for EKF).
        
        Uses autograd for automatic differentiation.
        """
        x_t = torch.from_numpy(x.astype(np.float32)).requires_grad_(True)
        
        # Compute output
        z = self.network(x_t)
        
        # Compute Jacobian row by row
        jacobian = []
        for i in range(self.n_obs):
            grad = torch.autograd.grad(z[i], x_t, retain_graph=True)[0]
            jacobian.append(grad.detach().numpy())
        
        return np.array(jacobian)


class DynamicsResidualNetwork(nn.Module):
    """
    Neural network that learns residual dynamics correction.
    
    Model: x_{t+1} = f_physics(x_t) + f_nn(x_t)
    
    The NN learns what physics-based model doesn't capture.
    """
    
    def __init__(
        self,
        n_states: int,
        hidden_dims: List[int] = None,
        activation: str = "tanh",
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 16]
        
        self.n_states = n_states
        
        layers = []
        in_dim = n_states
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "relu":
                layers.append(nn.ReLU())
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, n_states))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize to near-zero (start with physics model)
        with torch.no_grad():
            self.network[-1].weight.mul_(0.01)
            self.network[-1].bias.zero_()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: state -> residual correction."""
        return self.network(x)
    
    def predict_numpy(self, x: np.ndarray) -> np.ndarray:
        """Predict residual from numpy array."""
        with torch.no_grad():
            x_t = torch.from_numpy(x.astype(np.float32))
            if x_t.dim() == 1:
                x_t = x_t.unsqueeze(0)
            residual = self.network(x_t)
            return residual.squeeze(0).numpy()


@dataclass
class HybridEKFConfig:
    """Configuration for hybrid EKF with learned measurement."""
    n_states: int = 2
    n_obs: int = 1
    hidden_dims: List[int] = None
    learning_rate: float = 0.01
    n_epochs: int = 100
    batch_size: int = 32
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [32, 16]


class HybridEKF:
    """
    Extended Kalman Filter with learned measurement function.
    
    Uses a neural network to model complex/unknown measurement function,
    combined with physics-based dynamics model.
    
    Workflow:
        1. Train measurement network on (state, observation) pairs
        2. Use trained network as h() in EKF
        3. Compute Jacobian via autograd for covariance update
    
    Example:
        >>> # Setup hybrid EKF
        >>> hybrid = HybridEKF(f_dynamics, F_jacobian, Q, R, config)
        >>> 
        >>> # Train measurement model
        >>> hybrid.train_measurement_model(X_train, Z_train)
        >>> 
        >>> # Run filtering
        >>> hybrid.initialize(x0, P0)
        >>> for z in observations:
        ...     hybrid.predict()
        ...     hybrid.update(z)
    """
    
    def __init__(
        self,
        f: Callable,           # Dynamics function
        F_jacobian: Callable,  # Dynamics Jacobian
        Q: np.ndarray,         # Process noise
        R: np.ndarray,         # Measurement noise
        config: HybridEKFConfig,
    ):
        self.f = f
        self.F_jacobian = F_jacobian
        self.Q = Q
        self.R = R
        self.config = config
        
        # Create measurement network
        self.h_network = MeasurementNetwork(
            n_states=config.n_states,
            n_obs=config.n_obs,
            hidden_dims=config.hidden_dims,
        )
        
        self.x: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None
        self.training_loss_history: List[float] = []
        
    def train_measurement_model(
        self,
        X_train: np.ndarray,
        Z_train: np.ndarray,
        verbose: bool = False,
    ) -> List[float]:
        """
        Train measurement network on (state, observation) pairs.
        
        Args:
            X_train: States (n_samples, n_states)
            Z_train: Observations (n_samples, n_obs)
            verbose: Print training progress
            
        Returns:
            loss_history: Training loss per epoch
        """
        # Convert to tensors
        X_t = torch.from_numpy(X_train.astype(np.float32))
        Z_t = torch.from_numpy(Z_train.astype(np.float32))
        
        if Z_t.dim() == 1:
            Z_t = Z_t.unsqueeze(1)
        
        # Setup optimizer
        optimizer = torch.optim.Adam(
            self.h_network.parameters(),
            lr=self.config.learning_rate
        )
        criterion = nn.MSELoss()
        
        # Training loop
        n_samples = X_t.shape[0]
        loss_history = []
        
        for epoch in range(self.config.n_epochs):
            # Shuffle data
            perm = torch.randperm(n_samples)
            X_shuffled = X_t[perm]
            Z_shuffled = Z_t[perm]
            
            epoch_loss = 0.0
            n_batches = 0
            
            for i in range(0, n_samples, self.config.batch_size):
                X_batch = X_shuffled[i:i+self.config.batch_size]
                Z_batch = Z_shuffled[i:i+self.config.batch_size]
                
                optimizer.zero_grad()
                Z_pred = self.h_network(X_batch)
                loss = criterion(Z_pred, Z_batch)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            loss_history.append(avg_loss)
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{self.config.n_epochs}: Loss = {avg_loss:.6f}")
        
        self.training_loss_history = loss_history
        return loss_history
    
    def initialize(self, x0: np.ndarray, P0: np.ndarray) -> None:
        """Initialize filter state."""
        self.x = x0.copy()
        self.P = P0.copy()
        
    def predict(self, u: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Prediction step (uses physics-based dynamics)."""
        if self.x is None:
            raise ValueError("Filter not initialized")
        
        # Nonlinear prediction
        x_pred = self.f(self.x, u)
        
        # Linearize
        F = self.F_jacobian(self.x)
        P_pred = F @ self.P @ F.T + self.Q
        
        self.x = x_pred
        self.P = P_pred
        
        return x_pred, P_pred
    
    def update(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Update step (uses learned measurement function)."""
        if self.x is None:
            raise ValueError("Filter not initialized")
        
        # Predicted measurement (NN)
        z_pred = self.h_network.predict_numpy(self.x)
        
        # Jacobian via autograd
        H = self.h_network.jacobian_numpy(self.x)
        
        # Innovation
        y = z - z_pred
        
        # Kalman gain
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update
        self.x = self.x + K @ y
        I_KH = np.eye(self.config.n_states) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        
        return self.x.copy(), self.P.copy()
    
    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current state estimate."""
        return self.x.copy(), self.P.copy()


class HybridParticleFilter:
    """
    Particle Filter with learned measurement likelihood.
    
    Uses neural network to model complex measurement function,
    enabling likelihood computation for arbitrary observations.
    """
    
    def __init__(
        self,
        f: Callable,           # Stochastic dynamics (x, u, rng) -> x_next
        h_network: MeasurementNetwork,  # Learned measurement
        R: np.ndarray,         # Measurement noise covariance
        n_particles: int = 1000,
    ):
        self.f = f
        self.h_network = h_network
        self.R = R
        self.n_particles = n_particles
        
        self.R_inv = np.linalg.inv(R)
        self.R_det = np.linalg.det(R)
        self.n_obs = R.shape[0]
        
        self.particles: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None
        self.rng: Optional[np.random.Generator] = None
        
    def initialize(
        self,
        mean: np.ndarray,
        cov: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        """Initialize particles."""
        self.rng = rng
        self.n_states = mean.shape[0]
        self.particles = rng.multivariate_normal(mean, cov, size=self.n_particles)
        self.weights = np.ones(self.n_particles) / self.n_particles
        
    def predict(self, u: Optional[np.ndarray] = None) -> np.ndarray:
        """Propagate particles through dynamics."""
        for i in range(self.n_particles):
            self.particles[i] = self.f(self.particles[i], u, self.rng)
        return self.particles
    
    def _gaussian_likelihood(self, z_obs: np.ndarray, z_pred: np.ndarray) -> float:
        """Compute Gaussian likelihood."""
        diff = z_obs - z_pred
        exponent = -0.5 * diff.T @ self.R_inv @ diff
        norm_const = 1.0 / np.sqrt((2 * np.pi)**self.n_obs * self.R_det)
        return norm_const * np.exp(exponent)
    
    def update(self, z: np.ndarray) -> np.ndarray:
        """Update weights using learned measurement model."""
        for i in range(self.n_particles):
            z_pred = self.h_network.predict_numpy(self.particles[i])
            self.weights[i] *= self._gaussian_likelihood(z, z_pred)
        
        # Normalize
        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights /= weight_sum
        else:
            self.weights = np.ones(self.n_particles) / self.n_particles
        
        return self.weights
    
    def resample(self) -> None:
        """Systematic resampling."""
        n_eff = 1.0 / np.sum(self.weights**2)
        if n_eff > 0.5 * self.n_particles:
            return
        
        cumsum = np.cumsum(self.weights)
        u0 = self.rng.uniform(0, 1.0 / self.n_particles)
        u = u0 + np.arange(self.n_particles) / self.n_particles
        indices = np.searchsorted(cumsum, u)
        self.particles = self.particles[indices].copy()
        self.weights = np.ones(self.n_particles) / self.n_particles
        
    def get_state_estimate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get weighted mean and covariance."""
        mean = np.sum(self.particles * self.weights[:, np.newaxis], axis=0)
        diff = self.particles - mean
        cov = (self.weights[:, np.newaxis, np.newaxis] * 
               diff[:, :, np.newaxis] * diff[:, np.newaxis, :]).sum(axis=0)
        return mean, cov


def create_nonlinear_measurement_system(
    complexity: str = "moderate"
) -> Tuple[Callable, Callable, Callable]:
    """
    Create a system with nonlinear measurement function.
    
    The measurement function is intentionally complex to demonstrate
    when learned models help.
    
    Args:
        complexity: "simple", "moderate", or "complex"
        
    Returns:
        h_true: True measurement function
        h_linear: Linear approximation (for comparison)
        h_jacobian_true: True Jacobian
    """
    if complexity == "simple":
        # Mildly nonlinear: h(x) = x[0] + 0.1*sin(x[0])
        def h_true(x):
            return np.array([x[0] + 0.1 * np.sin(x[0])])
        
        def h_linear(x):
            return np.array([x[0]])
        
        def h_jacobian_true(x):
            return np.array([[1 + 0.1 * np.cos(x[0]), 0]])
            
    elif complexity == "moderate":
        # Moderately nonlinear: h(x) = sin(x[0]) + 0.5*x[0]*x[1]
        def h_true(x):
            return np.array([np.sin(x[0]) + 0.5 * x[0] * x[1]])
        
        def h_linear(x):
            return np.array([x[0]])  # Poor linear approximation
        
        def h_jacobian_true(x):
            return np.array([[np.cos(x[0]) + 0.5 * x[1], 0.5 * x[0]]])
            
    elif complexity == "complex":
        # Highly nonlinear: h(x) = tanh(x[0]) * exp(-0.1*x[1]^2)
        def h_true(x):
            return np.array([np.tanh(x[0]) * np.exp(-0.1 * x[1]**2)])
        
        def h_linear(x):
            return np.array([x[0]])  # Very poor approximation
        
        def h_jacobian_true(x):
            sech2 = 1 / np.cosh(x[0])**2
            exp_term = np.exp(-0.1 * x[1]**2)
            return np.array([
                [sech2 * exp_term, 
                 np.tanh(x[0]) * (-0.2 * x[1]) * exp_term]
            ])
    else:
        raise ValueError(f"Unknown complexity: {complexity}")
    
    return h_true, h_linear, h_jacobian_true


def generate_training_data(
    f_dynamics: Callable,
    h_true: Callable,
    n_samples: int,
    noise_std: float,
    rng: np.random.Generator,
    x_range: Tuple[float, float] = (-2, 2),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate training data for measurement network.
    
    Samples states uniformly and computes noisy measurements.
    
    Args:
        f_dynamics: Not used (for interface consistency)
        h_true: True measurement function
        n_samples: Number of samples
        noise_std: Measurement noise standard deviation
        rng: Random number generator
        x_range: Range for state sampling
        
    Returns:
        X: States (n_samples, n_states)
        Z: Noisy measurements (n_samples, n_obs)
    """
    n_states = 2  # Hardcoded for simplicity
    
    # Sample states uniformly
    X = rng.uniform(x_range[0], x_range[1], size=(n_samples, n_states))
    
    # Compute measurements with noise
    Z = np.array([h_true(x) + rng.normal(0, noise_std, size=1) for x in X])
    
    return X, Z.squeeze()

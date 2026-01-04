"""Constrained learning with conservation laws."""

import torch
import torch.nn as nn
import numpy as np
from typing import Callable, Tuple, Optional
from dataclasses import dataclass

from .pinn_base import PINN, PINNConfig


@dataclass
class ConservationConfig:
    """Configuration for conservation-constrained learning."""
    
    # Model architecture
    input_dim: int = 2  # e.g., [position, velocity]
    output_dim: int = 1  # e.g., next position
    hidden_dims: list[int] = None
    
    # Training
    n_train: int = 1000
    epochs: int = 2000
    lr: float = 1e-3
    batch_size: int = 64
    
    # Conservation penalty
    lambda_conservation: float = 1.0
    conservation_type: str = "energy"  # energy, momentum, mass
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 64]


class ConservationConstrainedNN:
    """
    Neural network with conservation law constraints.
    
    Trains a model while penalizing violations of physical conservation laws
    (energy, momentum, or mass).
    
    Example:
        >>> config = ConservationConfig(conservation_type="energy")
        >>> model = ConservationConstrainedNN(config)
        >>> X_train, y_train = generate_pendulum_data(n=1000)
        >>> history = model.train(X_train, y_train)
    """
    
    def __init__(self, config: ConservationConfig):
        self.config = config
        
        # Build network
        pinn_config = PINNConfig(
            input_dim=config.input_dim,
            output_dim=config.output_dim,
            hidden_dims=config.hidden_dims,
            activation="tanh",
        )
        self.model = PINN(pinn_config)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
    
    def energy_conservation_penalty(
        self,
        x_input: torch.Tensor,
        y_pred: torch.Tensor,
        mass: float = 1.0,
        spring_k: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute energy conservation penalty for mechanical system.
        
        For harmonic oscillator: E = 0.5 * m * v² + 0.5 * k * x²
        
        Args:
            x_input: Input [batch, 2] where [:, 0] = position, [:, 1] = velocity
            y_pred: Predicted next state [batch, 2]
            mass: Mass parameter
            spring_k: Spring constant
            
        Returns:
            Energy violation penalty
        """
        # Current state
        x_curr = x_input[:, 0:1]
        v_curr = x_input[:, 1:2]
        
        # Next state
        x_next = y_pred[:, 0:1]
        v_next = y_pred[:, 1:2]
        
        # Energy at current and next state
        E_curr = 0.5 * mass * v_curr ** 2 + 0.5 * spring_k * x_curr ** 2
        E_next = 0.5 * mass * v_next ** 2 + 0.5 * spring_k * x_next ** 2
        
        # Energy should be conserved
        energy_violation = torch.mean((E_next - E_curr) ** 2)
        
        return energy_violation
    
    def momentum_conservation_penalty(
        self,
        x_input: torch.Tensor,
        y_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute momentum conservation penalty.
        
        For isolated system: Σ m_i * v_i = constant
        
        Args:
            x_input: Input state [batch, n_particles * 2] (positions and velocities)
            y_pred: Predicted next state [batch, n_particles * 2]
            
        Returns:
            Momentum violation penalty
        """
        # Assume uniform mass = 1
        # Sum velocities (every other column starting from 1)
        v_curr = x_input[:, 1::2]  # Extract velocities
        v_next = y_pred[:, 1::2]
        
        p_curr = torch.sum(v_curr, dim=1)
        p_next = torch.sum(v_next, dim=1)
        
        momentum_violation = torch.mean((p_next - p_curr) ** 2)
        
        return momentum_violation
    
    def mass_conservation_penalty(
        self,
        x_input: torch.Tensor,
        y_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute mass conservation penalty for diffusion/reaction systems.
        
        For mass: Σ u_i = constant (no creation/destruction)
        
        Args:
            x_input: Current state [batch, n_cells]
            y_pred: Predicted next state [batch, n_cells]
            
        Returns:
            Mass violation penalty
        """
        mass_curr = torch.sum(x_input, dim=1)
        mass_next = torch.sum(y_pred, dim=1)
        
        mass_violation = torch.mean((mass_next - mass_curr) ** 2)
        
        return mass_violation
    
    def compute_conservation_penalty(
        self,
        x_input: torch.Tensor,
        y_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute conservation penalty based on config.
        
        Args:
            x_input: Input state
            y_pred: Predicted state
            
        Returns:
            Conservation penalty
        """
        if self.config.conservation_type == "energy":
            return self.energy_conservation_penalty(x_input, y_pred)
        elif self.config.conservation_type == "momentum":
            return self.momentum_conservation_penalty(x_input, y_pred)
        elif self.config.conservation_type == "mass":
            return self.mass_conservation_penalty(x_input, y_pred)
        else:
            raise ValueError(f"Unknown conservation type: {self.config.conservation_type}")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: int = 100,
    ) -> dict:
        """
        Train model with conservation constraints.
        
        Args:
            X_train: Training inputs [N, input_dim]
            y_train: Training outputs [N, output_dim]
            X_val: Validation inputs (optional)
            y_val: Validation outputs (optional)
            verbose: Print frequency
            
        Returns:
            history: Training history
        """
        # Convert to tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        
        if X_val is not None:
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
        
        history = {
            'loss': [],
            'loss_mse': [],
            'loss_conservation': [],
            'val_loss': [] if X_val is not None else None,
        }
        
        self.model.train()
        
        # Training loop
        for epoch in range(self.config.epochs):
            # Mini-batch training
            indices = torch.randperm(len(X_train_tensor))[:self.config.batch_size]
            X_batch = X_train_tensor[indices]
            y_batch = y_train_tensor[indices]
            
            self.optimizer.zero_grad()
            
            # Forward pass
            y_pred = self.model(X_batch)
            
            # MSE loss
            loss_mse = torch.mean((y_pred - y_batch) ** 2)
            
            # Conservation penalty
            loss_conservation = self.compute_conservation_penalty(X_batch, y_pred)
            
            # Total loss
            total_loss = loss_mse + self.config.lambda_conservation * loss_conservation
            
            total_loss.backward()
            self.optimizer.step()
            
            # Record history
            history['loss'].append(total_loss.item())
            history['loss_mse'].append(loss_mse.item())
            history['loss_conservation'].append(loss_conservation.item())
            
            # Validation
            if X_val is not None and (epoch + 1) % verbose == 0:
                self.model.eval()
                with torch.no_grad():
                    y_val_pred = self.model(X_val_tensor)
                    val_loss = torch.mean((y_val_pred - y_val_tensor) ** 2)
                    history['val_loss'].append(val_loss.item())
                self.model.train()
            
            # Verbose logging
            if verbose > 0 and (epoch + 1) % verbose == 0:
                log_str = (f"Epoch {epoch+1}/{self.config.epochs} - "
                          f"Loss: {total_loss.item():.6f} "
                          f"(MSE: {loss_mse.item():.6f}, "
                          f"Conservation: {loss_conservation.item():.6f})")
                
                if X_val is not None:
                    log_str += f", Val: {val_loss.item():.6f}"
                
                print(log_str)
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict outputs for given inputs.
        
        Args:
            X: Input array [N, input_dim]
            
        Returns:
            y: Predictions [N, output_dim]
        """
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = self.model(X_tensor)
            y = y_tensor.numpy()
        
        return y


def generate_pendulum_data(
    n: int,
    omega: float = 1.0,
    dt: float = 0.1,
    noise_std: float = 0.01,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic pendulum trajectory data.
    
    Ground truth follows: x(t+dt) = x(t) + v(t)*dt
                         v(t+dt) = v(t) - ω²*x(t)*dt
    
    Args:
        n: Number of samples
        omega: Angular frequency
        dt: Time step
        noise_std: Observation noise
        seed: Random seed
        
    Returns:
        X: Current state [n, 2] (position, velocity)
        y: Next state [n, 2]
    """
    np.random.seed(seed)
    
    # Generate random initial conditions
    x0 = np.random.randn(n) * 0.5
    v0 = np.random.randn(n) * 0.5
    
    # Current state
    X = np.stack([x0, v0], axis=1)
    
    # Next state (Euler integration)
    x1 = x0 + v0 * dt
    v1 = v0 - omega**2 * x0 * dt
    
    y = np.stack([x1, v1], axis=1)
    
    # Add noise
    y += np.random.randn(*y.shape) * noise_std
    
    return X, y


def compute_energy_violation(
    X: np.ndarray,
    y_pred: np.ndarray,
    mass: float = 1.0,
    spring_k: float = 1.0,
) -> np.ndarray:
    """
    Compute energy conservation violation.
    
    Args:
        X: Current state [n, 2]
        y_pred: Predicted next state [n, 2]
        mass: Mass
        spring_k: Spring constant
        
    Returns:
        Energy violations [n]
    """
    x_curr, v_curr = X[:, 0], X[:, 1]
    x_next, v_next = y_pred[:, 0], y_pred[:, 1]
    
    E_curr = 0.5 * mass * v_curr ** 2 + 0.5 * spring_k * x_curr ** 2
    E_next = 0.5 * mass * v_next ** 2 + 0.5 * spring_k * x_next ** 2
    
    violations = np.abs(E_next - E_curr)
    
    return violations

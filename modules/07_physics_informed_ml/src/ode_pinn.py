"""ODE solver using Physics-Informed Neural Networks."""

import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, Callable, Optional
from dataclasses import dataclass

from .pinn_base import PINN, PINNConfig, compute_gradient, train_pinn


@dataclass
class HarmonicOscillatorConfig:
    """Configuration for harmonic oscillator ODE."""
    
    omega: float = 1.0  # Angular frequency
    x0: float = 1.0  # Initial position
    v0: float = 0.0  # Initial velocity
    t_max: float = 10.0  # Integration time
    
    # PINN training
    n_collocation: int = 200  # Number of collocation points
    hidden_dims: list[int] = None
    epochs: int = 5000
    lr: float = 1e-3
    lambda_physics: float = 1.0
    lambda_ic: float = 10.0  # Higher weight for initial conditions
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [32, 32, 32]


class HarmonicOscillatorPINN:
    """
    PINN for solving harmonic oscillator ODE: d²x/dt² + ω²x = 0.
    
    Physics constraint: acceleration = -ω² * position
    Initial conditions: x(0) = x0, dx/dt(0) = v0
    
    Example:
        >>> config = HarmonicOscillatorConfig(omega=2.0, x0=1.0, v0=0.0)
        >>> pinn = HarmonicOscillatorPINN(config)
        >>> history = pinn.train(verbose=500)
        >>> t_test = np.linspace(0, 10, 100)
        >>> x_pred = pinn.predict(t_test)
    """
    
    def __init__(self, config: HarmonicOscillatorConfig):
        self.config = config
        
        # Build PINN (1D input: time, 1D output: position)
        pinn_config = PINNConfig(
            input_dim=1,
            output_dim=1,
            hidden_dims=config.hidden_dims,
            activation="tanh",
        )
        self.model = PINN(pinn_config)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        
        # Prepare training data
        self._prepare_training_points()
    
    def _prepare_training_points(self):
        """Prepare collocation points and initial conditions."""
        config = self.config
        
        # Collocation points (interior domain)
        t_physics = np.linspace(0, config.t_max, config.n_collocation)
        self.t_physics = torch.tensor(t_physics, dtype=torch.float32).view(-1, 1)
        self.t_physics.requires_grad = True
        
        # Initial condition points
        t_ic = torch.tensor([[0.0]], dtype=torch.float32, requires_grad=True)
        self.t_ic = t_ic
        
        # Initial values (position and velocity)
        self.x_ic = torch.tensor([[config.x0]], dtype=torch.float32)
        self.v_ic = torch.tensor([[config.v0]], dtype=torch.float32)
    
    def physics_residual(self, model: PINN, t: torch.Tensor) -> torch.Tensor:
        """
        Compute ODE residual: d²x/dt² + ω²x.
        
        Args:
            model: PINN model
            t: Time points [N, 1]
            
        Returns:
            Residual [N, 1]
        """
        x = model(t)
        
        # First derivative: velocity
        v = compute_gradient(x, t, order=1)
        
        # Second derivative: acceleration
        a = compute_gradient(x, t, order=2)
        
        # Physics: a + ω²x = 0
        omega_sq = self.config.omega ** 2
        residual = a + omega_sq * x
        
        return residual
    
    def initial_condition_loss(self) -> Tuple[torch.Tensor, dict]:
        """
        Compute initial condition loss.
        
        Returns:
            loss: IC loss
            loss_dict: Dictionary with position and velocity IC losses
        """
        # Position IC: x(0) = x0
        x_pred = self.model(self.t_ic)
        loss_x = torch.mean((x_pred - self.x_ic) ** 2)
        
        # Velocity IC: dx/dt(0) = v0
        v_pred = compute_gradient(x_pred, self.t_ic, order=1)
        loss_v = torch.mean((v_pred - self.v_ic) ** 2)
        
        loss = loss_x + loss_v
        
        loss_dict = {
            'ic_position': loss_x.item(),
            'ic_velocity': loss_v.item(),
        }
        
        return loss, loss_dict
    
    def train(
        self,
        verbose: int = 100,
        early_stopping_patience: Optional[int] = 500,
    ) -> dict:
        """
        Train PINN to solve ODE.
        
        Args:
            verbose: Print frequency (0 to disable)
            early_stopping_patience: Stop if no improvement for N epochs
            
        Returns:
            history: Training history
        """
        history = {
            'loss': [],
            'loss_physics': [],
            'loss_ic': [],
        }
        
        best_loss = float('inf')
        patience_counter = 0
        
        self.model.train()
        
        for epoch in range(self.config.epochs):
            self.optimizer.zero_grad()
            
            # Physics loss
            residual = self.physics_residual(self.model, self.t_physics)
            loss_physics = torch.mean(residual ** 2)
            
            # Initial condition loss
            loss_ic, ic_dict = self.initial_condition_loss()
            
            # Total loss
            total_loss = (
                self.config.lambda_physics * loss_physics +
                self.config.lambda_ic * loss_ic
            )
            
            total_loss.backward()
            self.optimizer.step()
            
            # Record history
            history['loss'].append(total_loss.item())
            history['loss_physics'].append(loss_physics.item())
            history['loss_ic'].append(loss_ic.item())
            
            # Verbose logging
            if verbose > 0 and (epoch + 1) % verbose == 0:
                print(f"Epoch {epoch+1}/{self.config.epochs} - "
                      f"Loss: {total_loss.item():.6f} "
                      f"(physics: {loss_physics.item():.6f}, "
                      f"IC: {loss_ic.item():.6f})")
            
            # Early stopping
            if early_stopping_patience is not None:
                if total_loss.item() < best_loss - 1e-6:
                    best_loss = total_loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        return history
    
    def predict(self, t: np.ndarray) -> np.ndarray:
        """
        Predict position at given times.
        
        Args:
            t: Time array [N]
            
        Returns:
            x: Position array [N]
        """
        self.model.eval()
        
        with torch.no_grad():
            t_tensor = torch.tensor(t, dtype=torch.float32).view(-1, 1)
            x_tensor = self.model(t_tensor)
            x = x_tensor.numpy().flatten()
        
        return x
    
    def predict_with_velocity(self, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict position and velocity at given times.
        
        Args:
            t: Time array [N]
            
        Returns:
            x: Position array [N]
            v: Velocity array [N]
        """
        self.model.eval()
        
        t_tensor = torch.tensor(t, dtype=torch.float32).view(-1, 1)
        t_tensor.requires_grad = True
        
        x_tensor = self.model(t_tensor)
        v_tensor = compute_gradient(x_tensor, t_tensor, order=1)
        
        x = x_tensor.detach().numpy().flatten()
        v = v_tensor.detach().numpy().flatten()
        
        return x, v


def solve_harmonic_oscillator_scipy(
    omega: float,
    x0: float,
    v0: float,
    t_eval: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve harmonic oscillator using scipy.integrate.solve_ivp.
    
    Baseline comparison for PINN.
    
    Args:
        omega: Angular frequency
        x0: Initial position
        v0: Initial velocity
        t_eval: Time points to evaluate
        
    Returns:
        x: Position array
        v: Velocity array
    """
    def harmonic_oscillator_ode(t, y):
        """ODE: dy/dt = [v, -ω²x]"""
        x, v = y
        dxdt = v
        dvdt = -omega**2 * x
        return [dxdt, dvdt]
    
    # Initial state
    y0 = [x0, v0]
    
    # Solve ODE
    sol = solve_ivp(
        harmonic_oscillator_ode,
        t_span=(t_eval[0], t_eval[-1]),
        y0=y0,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-8,
        atol=1e-10,
    )
    
    x = sol.y[0]
    v = sol.y[1]
    
    return x, v


def analytical_harmonic_oscillator(
    omega: float,
    x0: float,
    v0: float,
    t: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analytical solution for harmonic oscillator.
    
    x(t) = x0 * cos(ωt) + (v0/ω) * sin(ωt)
    v(t) = -x0 * ω * sin(ωt) + v0 * cos(ωt)
    
    Args:
        omega: Angular frequency
        x0: Initial position
        v0: Initial velocity
        t: Time array
        
    Returns:
        x: Position
        v: Velocity
    """
    x = x0 * np.cos(omega * t) + (v0 / omega) * np.sin(omega * t)
    v = -x0 * omega * np.sin(omega * t) + v0 * np.cos(omega * t)
    
    return x, v


def compute_energy(x: np.ndarray, v: np.ndarray, omega: float) -> np.ndarray:
    """
    Compute mechanical energy: E = 0.5 * (v² + ω²x²).
    
    For harmonic oscillator, energy should be conserved.
    
    Args:
        x: Position
        v: Velocity
        omega: Angular frequency
        
    Returns:
        E: Energy array
    """
    return 0.5 * (v**2 + omega**2 * x**2)

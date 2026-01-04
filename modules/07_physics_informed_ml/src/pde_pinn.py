"""PDE solver using Physics-Informed Neural Networks (1D Heat Equation)."""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

from .pinn_base import PINN, PINNConfig, compute_gradient, train_pinn


@dataclass
class HeatEquationConfig:
    """Configuration for 1D heat equation."""
    
    alpha: float = 0.01  # Thermal diffusivity
    x_min: float = 0.0
    x_max: float = 1.0
    t_max: float = 1.0
    
    # Boundary conditions (Dirichlet)
    bc_left: float = 0.0
    bc_right: float = 0.0
    
    # Initial condition function
    initial_condition: str = "gaussian"  # gaussian, sine, step
    
    # PINN training
    n_collocation_x: int = 50
    n_collocation_t: int = 50
    n_boundary: int = 50
    n_initial: int = 50
    
    hidden_dims: list[int] = None
    epochs: int = 10000
    lr: float = 1e-3
    lambda_physics: float = 1.0
    lambda_bc: float = 10.0
    lambda_ic: float = 10.0
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 64, 64, 64]


class HeatEquationPINN:
    """
    PINN for solving 1D heat equation: ∂u/∂t = α ∂²u/∂x².
    
    Domain: x ∈ [x_min, x_max], t ∈ [0, t_max]
    Boundary conditions: u(x_min, t) = bc_left, u(x_max, t) = bc_right
    Initial condition: u(x, 0) = u0(x)
    
    Example:
        >>> config = HeatEquationConfig(alpha=0.01, initial_condition="gaussian")
        >>> pinn = HeatEquationPINN(config)
        >>> history = pinn.train(verbose=1000)
        >>> x, t = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
        >>> u = pinn.predict(x.flatten(), t.flatten()).reshape(x.shape)
    """
    
    def __init__(self, config: HeatEquationConfig):
        self.config = config
        
        # Build PINN (2D input: [x, t], 1D output: u)
        pinn_config = PINNConfig(
            input_dim=2,
            output_dim=1,
            hidden_dims=config.hidden_dims,
            activation="tanh",
        )
        self.model = PINN(pinn_config)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        
        # Prepare training points
        self._prepare_training_points()
    
    def _initial_condition_fn(self, x: np.ndarray) -> np.ndarray:
        """Compute initial condition u(x, 0)."""
        config = self.config
        
        if config.initial_condition == "gaussian":
            # Gaussian bump
            center = (config.x_max + config.x_min) / 2
            width = (config.x_max - config.x_min) / 10
            u0 = np.exp(-((x - center) ** 2) / (2 * width ** 2))
        
        elif config.initial_condition == "sine":
            # Sine wave
            L = config.x_max - config.x_min
            u0 = np.sin(np.pi * (x - config.x_min) / L)
        
        elif config.initial_condition == "step":
            # Step function
            center = (config.x_max + config.x_min) / 2
            u0 = np.where(x < center, 1.0, 0.0)
        
        else:
            raise ValueError(f"Unknown initial condition: {config.initial_condition}")
        
        return u0
    
    def _prepare_training_points(self):
        """Prepare collocation, boundary, and initial condition points."""
        config = self.config
        
        # Collocation points (interior domain)
        x_col = np.random.uniform(config.x_min, config.x_max, config.n_collocation_x)
        t_col = np.random.uniform(0, config.t_max, config.n_collocation_t)
        x_col_grid, t_col_grid = np.meshgrid(x_col, t_col)
        
        self.xt_physics = torch.tensor(
            np.stack([x_col_grid.flatten(), t_col_grid.flatten()], axis=1),
            dtype=torch.float32,
        )
        self.xt_physics.requires_grad = True
        
        # Boundary points (x = x_min and x = x_max for all t)
        t_bc = np.linspace(0, config.t_max, config.n_boundary)
        
        # Left boundary
        x_bc_left = np.full_like(t_bc, config.x_min)
        xt_bc_left = np.stack([x_bc_left, t_bc], axis=1)
        u_bc_left = np.full_like(t_bc, config.bc_left)
        
        # Right boundary
        x_bc_right = np.full_like(t_bc, config.x_max)
        xt_bc_right = np.stack([x_bc_right, t_bc], axis=1)
        u_bc_right = np.full_like(t_bc, config.bc_right)
        
        # Combine boundaries
        self.xt_boundary = torch.tensor(
            np.vstack([xt_bc_left, xt_bc_right]),
            dtype=torch.float32,
        )
        self.u_boundary = torch.tensor(
            np.concatenate([u_bc_left, u_bc_right]),
            dtype=torch.float32,
        ).view(-1, 1)
        
        # Initial condition points (t = 0 for all x)
        x_ic = np.linspace(config.x_min, config.x_max, config.n_initial)
        t_ic = np.zeros_like(x_ic)
        
        self.xt_initial = torch.tensor(
            np.stack([x_ic, t_ic], axis=1),
            dtype=torch.float32,
        )
        
        u_ic = self._initial_condition_fn(x_ic)
        self.u_initial = torch.tensor(u_ic, dtype=torch.float32).view(-1, 1)
    
    def physics_residual(self, model: PINN, xt: torch.Tensor) -> torch.Tensor:
        """
        Compute heat equation residual: ∂u/∂t - α ∂²u/∂x².
        
        Args:
            model: PINN model
            xt: Space-time points [N, 2] where xt[:, 0] = x, xt[:, 1] = t
            
        Returns:
            Residual [N, 1]
        """
        u = model(xt)
        
        # Separate x and t
        x = xt[:, 0:1]
        t = xt[:, 1:2]
        
        # ∂u/∂t
        u_t = torch.autograd.grad(
            outputs=u,
            inputs=xt,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0][:, 1:2]  # Extract t component
        
        # ∂u/∂x
        u_x = torch.autograd.grad(
            outputs=u,
            inputs=xt,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0][:, 0:1]  # Extract x component
        
        # ∂²u/∂x²
        u_xx = torch.autograd.grad(
            outputs=u_x,
            inputs=xt,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True,
        )[0][:, 0:1]  # Extract x component
        
        # Heat equation: u_t - α * u_xx = 0
        residual = u_t - self.config.alpha * u_xx
        
        return residual
    
    def train(
        self,
        verbose: int = 100,
        early_stopping_patience: Optional[int] = 1000,
    ) -> dict:
        """
        Train PINN to solve heat equation.
        
        Args:
            verbose: Print frequency (0 to disable)
            early_stopping_patience: Stop if no improvement for N epochs
            
        Returns:
            history: Training history
        """
        history = {
            'loss': [],
            'loss_physics': [],
            'loss_bc': [],
            'loss_ic': [],
        }
        
        best_loss = float('inf')
        patience_counter = 0
        
        self.model.train()
        
        for epoch in range(self.config.epochs):
            self.optimizer.zero_grad()
            
            # Physics loss
            residual = self.physics_residual(self.model, self.xt_physics)
            loss_physics = torch.mean(residual ** 2)
            
            # Boundary condition loss
            u_bc_pred = self.model(self.xt_boundary)
            loss_bc = torch.mean((u_bc_pred - self.u_boundary) ** 2)
            
            # Initial condition loss
            u_ic_pred = self.model(self.xt_initial)
            loss_ic = torch.mean((u_ic_pred - self.u_initial) ** 2)
            
            # Total loss
            total_loss = (
                self.config.lambda_physics * loss_physics +
                self.config.lambda_bc * loss_bc +
                self.config.lambda_ic * loss_ic
            )
            
            total_loss.backward()
            self.optimizer.step()
            
            # Record history
            history['loss'].append(total_loss.item())
            history['loss_physics'].append(loss_physics.item())
            history['loss_bc'].append(loss_bc.item())
            history['loss_ic'].append(loss_ic.item())
            
            # Verbose logging
            if verbose > 0 and (epoch + 1) % verbose == 0:
                print(f"Epoch {epoch+1}/{self.config.epochs} - "
                      f"Loss: {total_loss.item():.6f} "
                      f"(physics: {loss_physics.item():.6f}, "
                      f"BC: {loss_bc.item():.6f}, "
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
    
    def predict(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Predict temperature at given space-time points.
        
        Args:
            x: Spatial coordinates [N]
            t: Time coordinates [N]
            
        Returns:
            u: Temperature [N]
        """
        self.model.eval()
        
        with torch.no_grad():
            xt_tensor = torch.tensor(
                np.stack([x, t], axis=1),
                dtype=torch.float32,
            )
            u_tensor = self.model(xt_tensor)
            u = u_tensor.numpy().flatten()
        
        return u


def solve_heat_equation_finite_difference(
    alpha: float,
    x: np.ndarray,
    t: np.ndarray,
    u0_fn: callable,
    bc_left: float = 0.0,
    bc_right: float = 0.0,
) -> np.ndarray:
    """
    Solve 1D heat equation using finite difference (Crank-Nicolson).
    
    Baseline comparison for PINN.
    
    Args:
        alpha: Thermal diffusivity
        x: Spatial grid [Nx]
        t: Time grid [Nt]
        u0_fn: Initial condition function
        bc_left: Left boundary value
        bc_right: Right boundary value
        
    Returns:
        u: Solution [Nt, Nx]
    """
    Nx = len(x)
    Nt = len(t)
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    
    # CFL condition check
    r = alpha * dt / (dx ** 2)
    if r > 0.5:
        print(f"Warning: CFL condition r={r:.3f} > 0.5, may be unstable")
    
    # Initialize solution
    u = np.zeros((Nt, Nx))
    u[0, :] = u0_fn(x)
    u[:, 0] = bc_left
    u[:, -1] = bc_right
    
    # Crank-Nicolson coefficients
    # Thomas algorithm for tridiagonal system
    a = -r / 2
    b = 1 + r
    c = -r / 2
    
    # Build tridiagonal matrix (implicit step)
    diag = np.full(Nx - 2, b)
    upper = np.full(Nx - 3, c)
    lower = np.full(Nx - 3, a)
    
    # Time stepping
    for n in range(Nt - 1):
        # Right-hand side (explicit part)
        rhs = np.zeros(Nx - 2)
        rhs[0] = (r/2) * u[n, 0] + (1 - r) * u[n, 1] + (r/2) * u[n, 2] + (r/2) * u[n+1, 0]
        rhs[-1] = (r/2) * u[n, -3] + (1 - r) * u[n, -2] + (r/2) * u[n, -1] + (r/2) * u[n+1, -1]
        
        for i in range(1, Nx - 3):
            rhs[i] = (r/2) * u[n, i] + (1 - r) * u[n, i+1] + (r/2) * u[n, i+2]
        
        # Solve tridiagonal system (Thomas algorithm)
        u[n+1, 1:-1] = thomas_algorithm(lower, diag, upper, rhs)
    
    return u


def thomas_algorithm(lower, diag, upper, rhs):
    """Solve tridiagonal system using Thomas algorithm."""
    n = len(diag)
    
    # Forward elimination
    c_star = np.zeros(n - 1)
    d_star = np.zeros(n)
    
    c_star[0] = upper[0] / diag[0]
    d_star[0] = rhs[0] / diag[0]
    
    for i in range(1, n - 1):
        denom = diag[i] - lower[i-1] * c_star[i-1]
        c_star[i] = upper[i] / denom
        d_star[i] = (rhs[i] - lower[i-1] * d_star[i-1]) / denom
    
    d_star[-1] = (rhs[-1] - lower[-1] * d_star[-2]) / (diag[-1] - lower[-1] * c_star[-2])
    
    # Back substitution
    x = np.zeros(n)
    x[-1] = d_star[-1]
    
    for i in range(n - 2, -1, -1):
        x[i] = d_star[i] - c_star[i] * x[i + 1]
    
    return x

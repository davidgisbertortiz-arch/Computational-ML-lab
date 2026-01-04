"""Base Physics-Informed Neural Network architecture."""

import torch
import torch.nn as nn
import numpy as np
from typing import Callable, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PINNConfig:
    """Configuration for PINN architecture."""
    
    input_dim: int = 1
    output_dim: int = 1
    hidden_dims: list[int] = None
    activation: str = "tanh"  # tanh, relu, silu
    use_batch_norm: bool = False
    dropout: float = 0.0
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [32, 32, 32]


class PINN(nn.Module):
    """
    Physics-Informed Neural Network base class.
    
    Flexible MLP architecture with automatic differentiation support
    for computing physics residuals.
    
    Example:
        >>> config = PINNConfig(input_dim=2, output_dim=1, hidden_dims=[64, 64])
        >>> pinn = PINN(config)
        >>> x = torch.randn(100, 2, requires_grad=True)
        >>> u = pinn(x)
        >>> u_x = compute_gradient(u, x, order=1)
    """
    
    def __init__(self, config: PINNConfig):
        super().__init__()
        self.config = config
        
        # Build MLP
        layers = []
        in_dim = config.input_dim
        
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            if config.activation == "tanh":
                layers.append(nn.Tanh())
            elif config.activation == "relu":
                layers.append(nn.ReLU())
            elif config.activation == "silu":
                layers.append(nn.SiLU())
            else:
                raise ValueError(f"Unknown activation: {config.activation}")
            
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            
            in_dim = hidden_dim
        
        # Output layer (no activation)
        layers.append(nn.Linear(in_dim, config.output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights (Xavier for tanh, He for ReLU/SiLU)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.config.activation == "tanh":
                    nn.init.xavier_normal_(m.weight)
                else:
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, input_dim]
            
        Returns:
            Output tensor [batch, output_dim]
        """
        return self.network(x)


def compute_gradient(
    output: torch.Tensor,
    input: torch.Tensor,
    order: int = 1,
    component_idx: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute gradient of output w.r.t. input using autodiff.
    
    Args:
        output: Network output [batch, output_dim]
        input: Network input [batch, input_dim] (requires_grad=True)
        order: Order of derivative (1 or 2)
        component_idx: If specified, compute gradient for this output component
        
    Returns:
        Gradient tensor [batch, input_dim] for order=1
        Second derivative tensor [batch, input_dim] for order=2
        
    Example:
        >>> x = torch.randn(100, 1, requires_grad=True)
        >>> u = model(x)
        >>> u_x = compute_gradient(u, x, order=1)
        >>> u_xx = compute_gradient(u, x, order=2)
    """
    if order not in [1, 2]:
        raise ValueError("Only order 1 and 2 derivatives supported")
    
    # Select component if multi-output
    if component_idx is not None:
        output = output[:, component_idx:component_idx+1]
    
    # First derivative
    grad = torch.autograd.grad(
        outputs=output,
        inputs=input,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    if order == 1:
        return grad
    
    # Second derivative
    grad2 = torch.autograd.grad(
        outputs=grad,
        inputs=input,
        grad_outputs=torch.ones_like(grad),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    return grad2


def pinn_loss(
    model: PINN,
    x_data: Optional[torch.Tensor],
    y_data: Optional[torch.Tensor],
    x_physics: torch.Tensor,
    physics_residual_fn: Callable,
    x_boundary: Optional[torch.Tensor] = None,
    boundary_values: Optional[torch.Tensor] = None,
    lambda_data: float = 1.0,
    lambda_physics: float = 1.0,
    lambda_boundary: float = 1.0,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute PINN loss: data + physics + boundary.
    
    Args:
        model: PINN model
        x_data: Data input points [N_data, input_dim] (optional)
        y_data: Data output values [N_data, output_dim] (optional)
        x_physics: Collocation points for physics [N_physics, input_dim]
        physics_residual_fn: Function computing PDE/ODE residual
        x_boundary: Boundary points [N_bc, input_dim] (optional)
        boundary_values: Boundary values [N_bc, output_dim] (optional)
        lambda_data: Weight for data loss
        lambda_physics: Weight for physics loss
        lambda_boundary: Weight for boundary loss
        
    Returns:
        total_loss: Combined loss
        loss_dict: Dictionary with individual loss components
        
    Example:
        >>> def heat_residual(model, x_t):
        ...     u = model(x_t)
        ...     u_t = compute_gradient(u, x_t[:, 1:2], order=1)
        ...     u_xx = compute_gradient(u, x_t[:, 0:1], order=2)
        ...     return u_t - alpha * u_xx
        >>> 
        >>> loss, losses = pinn_loss(model, None, None, x_physics, heat_residual)
    """
    loss_dict = {}
    total_loss = 0.0
    
    # Data loss (supervised)
    if x_data is not None and y_data is not None:
        y_pred = model(x_data)
        loss_data = torch.mean((y_pred - y_data) ** 2)
        loss_dict['data'] = loss_data.item()
        total_loss += lambda_data * loss_data
    else:
        loss_dict['data'] = 0.0
    
    # Physics loss (PDE/ODE residual)
    residual = physics_residual_fn(model, x_physics)
    loss_physics = torch.mean(residual ** 2)
    loss_dict['physics'] = loss_physics.item()
    total_loss += lambda_physics * loss_physics
    
    # Boundary/initial condition loss
    if x_boundary is not None and boundary_values is not None:
        bc_pred = model(x_boundary)
        loss_boundary = torch.mean((bc_pred - boundary_values) ** 2)
        loss_dict['boundary'] = loss_boundary.item()
        total_loss += lambda_boundary * loss_boundary
    else:
        loss_dict['boundary'] = 0.0
    
    loss_dict['total'] = total_loss.item()
    
    return total_loss, loss_dict


def train_pinn(
    model: PINN,
    optimizer: torch.optim.Optimizer,
    x_data: Optional[torch.Tensor],
    y_data: Optional[torch.Tensor],
    x_physics: torch.Tensor,
    physics_residual_fn: Callable,
    x_boundary: Optional[torch.Tensor] = None,
    boundary_values: Optional[torch.Tensor] = None,
    lambda_data: float = 1.0,
    lambda_physics: float = 1.0,
    lambda_boundary: float = 1.0,
    epochs: int = 1000,
    verbose: int = 100,
    early_stopping_patience: Optional[int] = None,
    early_stopping_delta: float = 1e-6,
) -> dict:
    """
    Train PINN with physics-informed loss.
    
    Args:
        model: PINN model
        optimizer: PyTorch optimizer
        x_data, y_data: Data (optional)
        x_physics: Collocation points
        physics_residual_fn: PDE/ODE residual function
        x_boundary, boundary_values: Boundary conditions (optional)
        lambda_*: Loss weights
        epochs: Number of training epochs
        verbose: Print frequency (0 to disable)
        early_stopping_patience: Stop if no improvement for N epochs
        early_stopping_delta: Minimum improvement threshold
        
    Returns:
        history: Dictionary with training history
    """
    history = {
        'loss': [],
        'loss_data': [],
        'loss_physics': [],
        'loss_boundary': [],
    }
    
    best_loss = float('inf')
    patience_counter = 0
    
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        loss, loss_dict = pinn_loss(
            model=model,
            x_data=x_data,
            y_data=y_data,
            x_physics=x_physics,
            physics_residual_fn=physics_residual_fn,
            x_boundary=x_boundary,
            boundary_values=boundary_values,
            lambda_data=lambda_data,
            lambda_physics=lambda_physics,
            lambda_boundary=lambda_boundary,
        )
        
        loss.backward()
        optimizer.step()
        
        # Record history
        history['loss'].append(loss_dict['total'])
        history['loss_data'].append(loss_dict['data'])
        history['loss_physics'].append(loss_dict['physics'])
        history['loss_boundary'].append(loss_dict['boundary'])
        
        # Verbose logging
        if verbose > 0 and (epoch + 1) % verbose == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss_dict['total']:.6f} "
                  f"(data: {loss_dict['data']:.6f}, "
                  f"physics: {loss_dict['physics']:.6f}, "
                  f"bc: {loss_dict['boundary']:.6f})")
        
        # Early stopping
        if early_stopping_patience is not None:
            if loss_dict['total'] < best_loss - early_stopping_delta:
                best_loss = loss_dict['total']
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return history

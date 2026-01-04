"""Extended Kalman Filter for nonlinear systems.

Handles nonlinear dynamics by linearizing around the current state estimate.
Good for weakly nonlinear systems with small noise.

References:
    Anderson & Moore (1979). "Optimal Filtering"
"""

import numpy as np
from typing import Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for nonlinear state-space models.
    
    Nonlinear model:
        x_t = f(x_{t-1}, u_t) + w_t,  w_t ~ N(0, Q)
        y_t = h(x_t) + v_t,            v_t ~ N(0, R)
    
    Linearization:
        F_t = ∂f/∂x |_{x_{t-1}}  (Jacobian of state transition)
        H_t = ∂h/∂x |_{x_t}      (Jacobian of observation)
    
    Args:
        f: Nonlinear state transition function (x, u) -> x_next
        h: Nonlinear observation function x -> z
        F_jacobian: Function returning Jacobian ∂f/∂x at x
        H_jacobian: Function returning Jacobian ∂h/∂x at x
        Q: Process noise covariance (n_states, n_states)
        R: Observation noise covariance (n_obs, n_obs)
        
    Example:
        >>> # Pendulum: state = [angle, angular_velocity]
        >>> def f(x, u):
        ...     theta, omega = x
        ...     g, L, dt = 9.81, 1.0, 0.1
        ...     omega_next = omega - (g/L) * np.sin(theta) * dt
        ...     theta_next = theta + omega_next * dt
        ...     return np.array([theta_next, omega_next])
        >>> 
        >>> def h(x):
        ...     return np.array([x[0]])  # observe angle only
        >>> 
        >>> def F_jac(x):
        ...     theta, omega = x
        ...     g, L, dt = 9.81, 1.0, 0.1
        ...     return np.array([
        ...         [1 - (g/L)*np.cos(theta)*dt**2, dt],
        ...         [-(g/L)*np.cos(theta)*dt, 1]
        ...     ])
        >>> 
        >>> def H_jac(x):
        ...     return np.array([[1, 0]])
        >>> 
        >>> Q = np.eye(2) * 0.01
        >>> R = np.array([[0.1]])
        >>> ekf = ExtendedKalmanFilter(f, h, F_jac, H_jac, Q, R)
    """
    
    f: Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]
    h: Callable[[np.ndarray], np.ndarray]
    F_jacobian: Callable[[np.ndarray], np.ndarray]
    H_jacobian: Callable[[np.ndarray], np.ndarray]
    Q: np.ndarray
    R: np.ndarray
    
    def __post_init__(self):
        """Initialize state variables."""
        self.n_states = self.Q.shape[0]
        self.n_obs = self.R.shape[0]
        
        assert self.Q.shape == (self.n_states, self.n_states), "Q shape mismatch"
        assert self.R.shape == (self.n_obs, self.n_obs), "R shape mismatch"
        
        self.x: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None
        
    def initialize(self, x0: np.ndarray, P0: np.ndarray) -> None:
        """
        Initialize filter with initial state and covariance.
        
        Args:
            x0: Initial state estimate (n_states,)
            P0: Initial state covariance (n_states, n_states)
        """
        assert x0.shape == (self.n_states,), "x0 shape mismatch"
        assert P0.shape == (self.n_states, self.n_states), "P0 shape mismatch"
        
        self.x = x0.copy()
        self.P = P0.copy()
        
    def predict(self, u: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediction step with nonlinear dynamics.
        
        Uses nonlinear f() for state prediction, but Jacobian F for covariance.
        
        Equations:
            x̂_{t|t-1} = f(x̂_{t-1|t-1}, u_t)
            F_t = ∂f/∂x |_{x̂_{t-1|t-1}}
            P_{t|t-1} = F_t P_{t-1|t-1} F_t^T + Q
        
        Args:
            u: Control input, optional
            
        Returns:
            x_pred: Predicted state (n_states,)
            P_pred: Predicted covariance (n_states, n_states)
        """
        if self.x is None or self.P is None:
            raise ValueError("Filter not initialized. Call initialize() first.")
        
        # Nonlinear prediction
        x_pred = self.f(self.x, u)
        
        # Linearize at current state
        F = self.F_jacobian(self.x)
        
        # Covariance prediction (using linearization)
        P_pred = F @ self.P @ F.T + self.Q
        
        self.x = x_pred
        self.P = P_pred
        
        return x_pred, P_pred
    
    def update(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update step with nonlinear observation.
        
        Uses nonlinear h() for innovation, but Jacobian H for gain computation.
        
        Equations:
            H_t = ∂h/∂x |_{x̂_{t|t-1}}
            Innovation: y = z - h(x̂_{t|t-1})
            Innovation covariance: S = H_t P_{t|t-1} H_t^T + R
            Kalman gain: K = P_{t|t-1} H_t^T S^{-1}
            x̂_{t|t} = x̂_{t|t-1} + K y
            P_{t|t} = (I - K H_t) P_{t|t-1}
        
        Args:
            z: Measurement (n_obs,)
            
        Returns:
            x_updated: Updated state (n_states,)
            P_updated: Updated covariance (n_states, n_states)
        """
        if self.x is None or self.P is None:
            raise ValueError("Filter not initialized.")
        
        assert z.shape == (self.n_obs,), f"Measurement shape mismatch: {z.shape}"
        
        # Predicted measurement (nonlinear)
        z_pred = self.h(self.x)
        
        # Innovation
        y = z - z_pred
        
        # Linearize observation at predicted state
        H = self.H_jacobian(self.x)
        
        # Innovation covariance
        S = H @ self.P @ H.T + self.R
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state
        x_updated = self.x + K @ y
        
        # Update covariance (Joseph form)
        I_KH = np.eye(self.n_states) - K @ H
        P_updated = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        
        self.x = x_updated
        self.P = P_updated
        
        return x_updated, P_updated
    
    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current state estimate and covariance."""
        if self.x is None or self.P is None:
            raise ValueError("Filter not initialized.")
        return self.x.copy(), self.P.copy()


def pendulum_dynamics(
    dt: float = 0.1,
    g: float = 9.81,
    L: float = 1.0
) -> Tuple[Callable, Callable]:
    """
    Create pendulum dynamics and Jacobian.
    
    State: [angle (rad), angular_velocity (rad/s)]
    Equation: θ̈ = -(g/L) sin(θ)
    
    Args:
        dt: Time step
        g: Gravitational acceleration
        L: Pendulum length
        
    Returns:
        f: State transition function
        F_jacobian: Jacobian function
    """
    def f(x: np.ndarray, u: Optional[np.ndarray] = None) -> np.ndarray:
        """Pendulum dynamics (Euler integration)."""
        theta, omega = x
        omega_next = omega - (g / L) * np.sin(theta) * dt
        theta_next = theta + omega_next * dt
        return np.array([theta_next, omega_next])
    
    def F_jacobian(x: np.ndarray) -> np.ndarray:
        """Jacobian of pendulum dynamics."""
        theta, omega = x
        # ∂f/∂x
        dtheta_next_dtheta = 1 - (g / L) * np.cos(theta) * dt**2
        dtheta_next_domega = dt
        domega_next_dtheta = -(g / L) * np.cos(theta) * dt
        domega_next_domega = 1
        
        return np.array([
            [dtheta_next_dtheta, dtheta_next_domega],
            [domega_next_dtheta, domega_next_domega]
        ])
    
    return f, F_jacobian


def angle_observation_model() -> Tuple[Callable, Callable]:
    """
    Observe angle only (not angular velocity).
    
    Returns:
        h: Observation function
        H_jacobian: Observation Jacobian
    """
    def h(x: np.ndarray) -> np.ndarray:
        """Observe angle."""
        return np.array([x[0]])
    
    def H_jacobian(x: np.ndarray) -> np.ndarray:
        """Jacobian of observation."""
        return np.array([[1, 0]])
    
    return h, H_jacobian

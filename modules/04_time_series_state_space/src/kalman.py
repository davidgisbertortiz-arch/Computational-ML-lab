"""Linear Kalman Filter for state estimation.

Implements the classic Kalman filter for linear Gaussian state-space models.
Optimal for systems with linear dynamics and Gaussian noise.

References:
    Kalman, R. E. (1960). "A New Approach to Linear Filtering and Prediction Problems"
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class KalmanFilter:
    """
    Linear Kalman Filter for optimal state estimation.
    
    State-space model:
        x_t = F x_{t-1} + B u_t + w_t,  w_t ~ N(0, Q)
        y_t = H x_t + v_t,               v_t ~ N(0, R)
    
    Args:
        F: State transition matrix (n_states, n_states)
        H: Observation matrix (n_obs, n_states)
        Q: Process noise covariance (n_states, n_states)
        R: Observation noise covariance (n_obs, n_obs)
        B: Control input matrix (n_states, n_control), optional
        
    Attributes:
        x: Current state estimate (n_states,)
        P: Current state covariance (n_states, n_states)
        
    Example:
        >>> # Constant velocity model: state = [position, velocity]
        >>> dt = 0.1
        >>> F = np.array([[1, dt], [0, 1]])  # position += velocity * dt
        >>> H = np.array([[1, 0]])           # observe position only
        >>> Q = np.eye(2) * 0.01             # small process noise
        >>> R = np.array([[1.0]])            # measurement noise
        >>> 
        >>> kf = KalmanFilter(F, H, Q, R)
        >>> kf.initialize(x0=np.array([0, 1]), P0=np.eye(2))
        >>> 
        >>> # Prediction step
        >>> kf.predict()
        >>> 
        >>> # Update with measurement
        >>> z = np.array([0.95])  # noisy position measurement
        >>> kf.update(z)
    """
    
    F: np.ndarray  # State transition matrix
    H: np.ndarray  # Observation matrix
    Q: np.ndarray  # Process noise covariance
    R: np.ndarray  # Observation noise covariance
    B: Optional[np.ndarray] = None  # Control matrix
    
    def __post_init__(self):
        """Validate dimensions and initialize state."""
        self.n_states = self.F.shape[0]
        self.n_obs = self.H.shape[0]
        
        # Validate shapes
        assert self.F.shape == (self.n_states, self.n_states), "F must be square"
        assert self.H.shape == (self.n_obs, self.n_states), "H shape mismatch"
        assert self.Q.shape == (self.n_states, self.n_states), "Q shape mismatch"
        assert self.R.shape == (self.n_obs, self.n_obs), "R shape mismatch"
        
        if self.B is not None:
            assert self.B.shape[0] == self.n_states, "B shape mismatch"
        
        # State estimate and covariance (initialized later)
        self.x: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None
        
    def initialize(self, x0: np.ndarray, P0: np.ndarray) -> None:
        """
        Initialize filter with initial state and covariance.
        
        Args:
            x0: Initial state estimate (n_states,)
            P0: Initial state covariance (n_states, n_states)
        """
        assert x0.shape == (self.n_states,), f"x0 shape mismatch: {x0.shape}"
        assert P0.shape == (self.n_states, self.n_states), "P0 shape mismatch"
        
        self.x = x0.copy()
        self.P = P0.copy()
        
    def predict(self, u: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediction step: propagate state and covariance forward.
        
        Equations:
            x̂_{t|t-1} = F x̂_{t-1|t-1} + B u_t
            P_{t|t-1} = F P_{t-1|t-1} F^T + Q
        
        Args:
            u: Control input (n_control,), optional
            
        Returns:
            x_pred: Predicted state (n_states,)
            P_pred: Predicted covariance (n_states, n_states)
        """
        if self.x is None or self.P is None:
            raise ValueError("Filter not initialized. Call initialize() first.")
        
        # Predict state
        x_pred = self.F @ self.x
        if u is not None and self.B is not None:
            x_pred += self.B @ u
            
        # Predict covariance
        P_pred = self.F @ self.P @ self.F.T + self.Q
        
        # Update internal state
        self.x = x_pred
        self.P = P_pred
        
        return x_pred, P_pred
    
    def update(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update step: correct prediction with measurement.
        
        Equations:
            Innovation: y = z - H x̂_{t|t-1}
            Innovation covariance: S = H P_{t|t-1} H^T + R
            Kalman gain: K = P_{t|t-1} H^T S^{-1}
            Updated state: x̂_{t|t} = x̂_{t|t-1} + K y
            Updated covariance: P_{t|t} = (I - K H) P_{t|t-1}
        
        Args:
            z: Measurement (n_obs,)
            
        Returns:
            x_updated: Updated state (n_states,)
            P_updated: Updated covariance (n_states, n_states)
        """
        if self.x is None or self.P is None:
            raise ValueError("Filter not initialized. Call initialize() first.")
        
        assert z.shape == (self.n_obs,), f"Measurement shape mismatch: {z.shape}"
        
        # Innovation (measurement residual)
        y = z - self.H @ self.x
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        x_updated = self.x + K @ y
        
        # Update covariance (Joseph form for numerical stability)
        I_KH = np.eye(self.n_states) - K @ self.H
        P_updated = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        
        # Update internal state
        self.x = x_updated
        self.P = P_updated
        
        return x_updated, P_updated
    
    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current state estimate and covariance.
        
        Returns:
            x: State estimate (n_states,)
            P: State covariance (n_states, n_states)
        """
        if self.x is None or self.P is None:
            raise ValueError("Filter not initialized.")
        return self.x.copy(), self.P.copy()


def constant_velocity_model(dt: float, process_noise: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a 1D constant velocity motion model.
    
    State: [position, velocity]
    Dynamics: position_{t+1} = position_t + velocity_t * dt
              velocity_{t+1} = velocity_t + noise
    
    Args:
        dt: Time step
        process_noise: Process noise standard deviation
        
    Returns:
        F: State transition matrix (2, 2)
        Q: Process noise covariance (2, 2)
    """
    F = np.array([
        [1, dt],
        [0, 1]
    ])
    
    # Process noise (continuous-time white noise acceleration model)
    Q = process_noise**2 * np.array([
        [dt**3/3, dt**2/2],
        [dt**2/2, dt]
    ])
    
    return F, Q


def position_observation_model(obs_noise: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Observe position only (not velocity).
    
    Args:
        obs_noise: Observation noise standard deviation
        
    Returns:
        H: Observation matrix (1, 2)
        R: Observation noise covariance (1, 1)
    """
    H = np.array([[1, 0]])
    R = np.array([[obs_noise**2]])
    return H, R

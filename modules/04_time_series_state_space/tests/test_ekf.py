"""Tests for Extended Kalman Filter implementation."""

import pytest
import numpy as np
from modules._import_helper import safe_import_from

ExtendedKalmanFilter, pendulum_dynamics, angle_observation_model = safe_import_from(
    '04_time_series_state_space.src.ekf',
    'ExtendedKalmanFilter', 'pendulum_dynamics', 'angle_observation_model'
)


class TestExtendedKalmanFilter:
    """Test Extended Kalman Filter implementation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.f, self.F_jac = pendulum_dynamics(dt=0.1)
        self.h, self.H_jac = angle_observation_model()
        self.Q = np.eye(2) * 0.01
        self.R = np.array([[0.1]])
        
    def test_initialization(self):
        """Test EKF initialization."""
        ekf = ExtendedKalmanFilter(
            self.f, self.h, self.F_jac, self.H_jac, self.Q, self.R
        )
        assert ekf.n_states == 2
        assert ekf.n_obs == 1
        
    def test_predict_shapes(self):
        """Test prediction step output shapes."""
        ekf = ExtendedKalmanFilter(
            self.f, self.h, self.F_jac, self.H_jac, self.Q, self.R
        )
        ekf.initialize(x0=np.array([0.1, 0.0]), P0=np.eye(2))
        
        x_pred, P_pred = ekf.predict()
        
        assert x_pred.shape == (2,)
        assert P_pred.shape == (2, 2)
        
    def test_update_shapes(self):
        """Test update step output shapes."""
        ekf = ExtendedKalmanFilter(
            self.f, self.h, self.F_jac, self.H_jac, self.Q, self.R
        )
        ekf.initialize(x0=np.array([0.1, 0.0]), P0=np.eye(2))
        ekf.predict()
        
        z = np.array([0.15])
        x_updated, P_updated = ekf.update(z)
        
        assert x_updated.shape == (2,)
        assert P_updated.shape == (2, 2)
        
    def test_pendulum_dynamics_small_angle(self):
        """Test pendulum dynamics for small angles (near linear)."""
        # For small angles, sin(θ) ≈ θ, so pendulum behaves linearly
        x0 = np.array([0.01, 0.0])  # Small angle, zero velocity
        
        x1 = self.f(x0, None)
        
        # Should oscillate (velocity should become negative)
        assert x1[1] < 0, "Angular velocity should decrease for positive angle"
        
    def test_jacobian_computation(self):
        """Test Jacobian computation."""
        x = np.array([0.5, 0.1])
        F = self.F_jac(x)
        
        assert F.shape == (2, 2)
        # Check that Jacobian is close to identity for small dt
        assert np.abs(F[0, 0] - 1.0) < 0.1
        assert np.abs(F[1, 1] - 1.0) < 0.1
        
    def test_reproducibility(self):
        """Test that EKF is deterministic given same inputs."""
        observations = np.array([0.1, 0.09, 0.07, 0.05])
        
        # Run 1
        ekf1 = ExtendedKalmanFilter(
            self.f, self.h, self.F_jac, self.H_jac, self.Q, self.R
        )
        ekf1.initialize(x0=np.array([0.1, 0.0]), P0=np.eye(2))
        states1 = []
        for z in observations:
            ekf1.predict()
            ekf1.update(np.array([z]))
            x, _ = ekf1.get_state()
            states1.append(x)
        
        # Run 2
        ekf2 = ExtendedKalmanFilter(
            self.f, self.h, self.F_jac, self.H_jac, self.Q, self.R
        )
        ekf2.initialize(x0=np.array([0.1, 0.0]), P0=np.eye(2))
        states2 = []
        for z in observations:
            ekf2.predict()
            ekf2.update(np.array([z]))
            x, _ = ekf2.get_state()
            states2.append(x)
        
        assert np.allclose(states1, states2)


class TestPendulumHelpers:
    """Test pendulum helper functions."""
    
    def test_pendulum_dynamics_creation(self):
        """Test pendulum dynamics function creation."""
        f, F_jac = pendulum_dynamics(dt=0.1, g=9.81, L=1.0)
        
        # Test function signature
        x = np.array([0.1, 0.0])
        x_next = f(x, None)
        assert x_next.shape == (2,)
        
        # Test Jacobian
        F = F_jac(x)
        assert F.shape == (2, 2)
        
    def test_angle_observation_model_creation(self):
        """Test angle observation model creation."""
        h, H_jac = angle_observation_model()
        
        x = np.array([0.5, 0.1])
        
        # Test observation function
        z = h(x)
        assert z.shape == (1,)
        assert z[0] == x[0]
        
        # Test Jacobian
        H = H_jac(x)
        assert H.shape == (1, 2)
        assert H[0, 0] == 1.0
        assert H[0, 1] == 0.0

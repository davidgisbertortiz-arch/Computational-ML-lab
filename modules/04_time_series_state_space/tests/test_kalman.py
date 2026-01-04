"""Tests for Kalman Filter implementation."""

import pytest
import numpy as np
from modules._import_helper import safe_import_from

KalmanFilter, constant_velocity_model, position_observation_model = safe_import_from(
    '04_time_series_state_space.src.kalman',
    'KalmanFilter', 'constant_velocity_model', 'position_observation_model'
)

set_seed = safe_import_from('00_repo_standards.src.mlphys_core', 'set_seed')


class TestKalmanFilter:
    """Test Kalman Filter implementation."""
    
    def test_initialization(self):
        """Test filter initialization."""
        F, Q = constant_velocity_model(dt=0.1)
        H, R = position_observation_model(obs_noise=1.0)
        
        kf = KalmanFilter(F, H, Q, R)
        assert kf.n_states == 2
        assert kf.n_obs == 1
        
    def test_predict_shapes(self):
        """Test prediction step output shapes."""
        F, Q = constant_velocity_model(dt=0.1)
        H, R = position_observation_model(obs_noise=1.0)
        
        kf = KalmanFilter(F, H, Q, R)
        kf.initialize(x0=np.array([0, 1]), P0=np.eye(2))
        
        x_pred, P_pred = kf.predict()
        
        assert x_pred.shape == (2,)
        assert P_pred.shape == (2, 2)
        assert np.allclose(P_pred, P_pred.T)  # Covariance is symmetric
        
    def test_update_shapes(self):
        """Test update step output shapes."""
        F, Q = constant_velocity_model(dt=0.1)
        H, R = position_observation_model(obs_noise=1.0)
        
        kf = KalmanFilter(F, H, Q, R)
        kf.initialize(x0=np.array([0, 1]), P0=np.eye(2))
        kf.predict()
        
        z = np.array([0.95])
        x_updated, P_updated = kf.update(z)
        
        assert x_updated.shape == (2,)
        assert P_updated.shape == (2, 2)
        
    def test_covariance_positive_definite(self):
        """Test that covariance remains positive definite."""
        F, Q = constant_velocity_model(dt=0.1)
        H, R = position_observation_model(obs_noise=1.0)
        
        kf = KalmanFilter(F, H, Q, R)
        kf.initialize(x0=np.array([0, 1]), P0=np.eye(2))
        
        # Run several steps
        for _ in range(10):
            kf.predict()
            z = np.array([np.random.randn()])
            kf.update(z)
            
            _, P = kf.get_state()
            eigenvalues = np.linalg.eigvals(P)
            assert np.all(eigenvalues > 0), "Covariance not positive definite"
            
    def test_reproducibility(self):
        """Test that filter is deterministic given same inputs."""
        F, Q = constant_velocity_model(dt=0.1)
        H, R = position_observation_model(obs_noise=1.0)
        
        observations = np.array([1.0, 1.1, 1.2, 1.3])
        
        # Run 1
        kf1 = KalmanFilter(F, H, Q, R)
        kf1.initialize(x0=np.array([0, 1]), P0=np.eye(2))
        states1 = []
        for z in observations:
            kf1.predict()
            kf1.update(np.array([z]))
            x, _ = kf1.get_state()
            states1.append(x)
        
        # Run 2
        kf2 = KalmanFilter(F, H, Q, R)
        kf2.initialize(x0=np.array([0, 1]), P0=np.eye(2))
        states2 = []
        for z in observations:
            kf2.predict()
            kf2.update(np.array([z]))
            x, _ = kf2.get_state()
            states2.append(x)
        
        assert np.allclose(states1, states2)
        
    def test_constant_velocity_tracking(self):
        """Sanity test: filter should track constant velocity motion."""
        set_seed(42)
        dt = 0.1
        F, Q = constant_velocity_model(dt=dt, process_noise=0.01)
        H, R = position_observation_model(obs_noise=0.1)
        
        # Simulate true constant velocity
        true_states = []
        observations = []
        x_true = np.array([0.0, 1.0])  # Initial: pos=0, vel=1
        
        for _ in range(50):
            x_true = F @ x_true  # No noise for simplicity
            true_states.append(x_true.copy())
            z = H @ x_true + np.random.randn(1) * 0.1
            observations.append(z[0])
        
        true_states = np.array(true_states)
        
        # Run filter
        kf = KalmanFilter(F, H, Q, R)
        kf.initialize(x0=np.array([0, 0]), P0=np.eye(2))
        
        estimated_states = []
        for z in observations:
            kf.predict()
            kf.update(np.array([z]))
            x, _ = kf.get_state()
            estimated_states.append(x)
        
        estimated_states = np.array(estimated_states)
        
        # Filter should converge close to true state
        final_error = np.abs(true_states[-1] - estimated_states[-1])
        assert final_error[0] < 0.5, "Position tracking too inaccurate"
        assert final_error[1] < 0.2, "Velocity estimation too inaccurate"


class TestHelperFunctions:
    """Test helper functions."""
    
    def test_constant_velocity_model(self):
        """Test constant velocity model creation."""
        dt = 0.1
        F, Q = constant_velocity_model(dt)
        
        assert F.shape == (2, 2)
        assert Q.shape == (2, 2)
        assert F[0, 1] == dt
        assert F[1, 1] == 1.0
        
    def test_position_observation_model(self):
        """Test position observation model creation."""
        H, R = position_observation_model(obs_noise=1.5)
        
        assert H.shape == (1, 2)
        assert R.shape == (1, 1)
        assert H[0, 0] == 1.0
        assert H[0, 1] == 0.0
        assert R[0, 0] == 1.5**2

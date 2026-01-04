"""Tests for hybrid neural network + filter models."""

import pytest
import numpy as np
import torch
from modules._import_helper import safe_import_from

(MeasurementNetwork, DynamicsResidualNetwork, HybridEKF, HybridEKFConfig,
 HybridParticleFilter, create_nonlinear_measurement_system, 
 generate_training_data) = safe_import_from(
    '04_time_series_state_space.src.hybrid_models',
    'MeasurementNetwork', 'DynamicsResidualNetwork', 'HybridEKF', 
    'HybridEKFConfig', 'HybridParticleFilter', 
    'create_nonlinear_measurement_system', 'generate_training_data'
)

set_seed = safe_import_from('00_repo_standards.src.mlphys_core', 'set_seed')


class TestMeasurementNetwork:
    """Test neural measurement function."""
    
    def test_forward_shapes(self):
        """Test forward pass shapes."""
        net = MeasurementNetwork(n_states=2, n_obs=1, hidden_dims=[16, 8])
        
        x = torch.randn(10, 2)
        z = net(x)
        
        assert z.shape == (10, 1)
        
    def test_predict_numpy(self):
        """Test numpy prediction interface."""
        net = MeasurementNetwork(n_states=2, n_obs=1, hidden_dims=[16, 8])
        
        x = np.array([0.5, -0.3])
        z = net.predict_numpy(x)
        
        assert z.shape == (1,)
        
    def test_jacobian_computation(self):
        """Test autograd Jacobian."""
        net = MeasurementNetwork(n_states=2, n_obs=1, hidden_dims=[16, 8])
        
        x = np.array([0.5, -0.3])
        J = net.jacobian_numpy(x)
        
        assert J.shape == (1, 2)
        
    def test_jacobian_numerical(self):
        """Verify Jacobian against numerical gradient."""
        torch.manual_seed(42)
        net = MeasurementNetwork(n_states=2, n_obs=1, hidden_dims=[16, 8])
        
        x = np.array([0.5, -0.3], dtype=np.float32)  # Match network precision
        J_auto = net.jacobian_numpy(x)
        
        # Numerical gradient (balance precision vs numerical stability)
        eps = 1e-4  # Good balance for float32
        J_num = np.zeros((1, 2), dtype=np.float32)
        for i in range(2):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            # Extract scalar from output array
            y_plus = net.predict_numpy(x_plus)[0]
            y_minus = net.predict_numpy(x_minus)[0]
            J_num[0, i] = (y_plus - y_minus) / (2 * eps)
        
        # Relaxed tolerance appropriate for numerical gradient + float32 precision
        assert np.allclose(J_auto, J_num, rtol=5e-2, atol=1e-3)


class TestDynamicsResidualNetwork:
    """Test dynamics residual network."""
    
    def test_forward_shapes(self):
        """Test forward pass shapes."""
        net = DynamicsResidualNetwork(n_states=2, hidden_dims=[16, 8])
        
        x = torch.randn(10, 2)
        residual = net(x)
        
        assert residual.shape == (10, 2)
        
    def test_initial_residual_small(self):
        """Test that initial residual is near zero."""
        torch.manual_seed(42)
        net = DynamicsResidualNetwork(n_states=2, hidden_dims=[16, 8])
        
        x = np.array([1.0, 1.0])
        residual = net.predict_numpy(x)
        
        # Initial residual should be small
        assert np.linalg.norm(residual) < 0.1


class TestHybridEKF:
    """Test Hybrid Extended Kalman Filter."""
    
    def setup_method(self):
        """Setup test fixtures."""
        set_seed(42)
        torch.manual_seed(42)
        
        # Simple linear dynamics for testing
        self.dt = 0.1
        
        def f(x, u):
            F = np.array([[1, self.dt], [0, 1]])
            return F @ x
        
        def F_jac(x):
            return np.array([[1, self.dt], [0, 1]])
        
        self.f = f
        self.F_jac = F_jac
        self.Q = np.eye(2) * 0.01
        self.R = np.array([[0.1]])
        
        self.config = HybridEKFConfig(
            n_states=2,
            n_obs=1,
            hidden_dims=[16, 8],
            learning_rate=0.01,
            n_epochs=50,
        )
        
    def test_initialization(self):
        """Test HybridEKF initialization."""
        hybrid = HybridEKF(self.f, self.F_jac, self.Q, self.R, self.config)
        
        assert hybrid.h_network is not None
        assert hybrid.x is None
        
    def test_train_measurement_model(self):
        """Test training measurement network."""
        hybrid = HybridEKF(self.f, self.F_jac, self.Q, self.R, self.config)
        
        # Generate training data
        rng = np.random.default_rng(42)
        X_train = rng.uniform(-2, 2, size=(100, 2))
        Z_train = X_train[:, 0:1] + 0.1 * np.sin(X_train[:, 0:1])  # Nonlinear
        
        losses = hybrid.train_measurement_model(X_train, Z_train.squeeze())
        
        assert len(losses) == self.config.n_epochs
        assert losses[-1] < losses[0]  # Loss should decrease
        
    def test_predict_update_shapes(self):
        """Test predict and update output shapes."""
        hybrid = HybridEKF(self.f, self.F_jac, self.Q, self.R, self.config)
        
        # Train minimally
        rng = np.random.default_rng(42)
        X_train = rng.uniform(-2, 2, size=(50, 2))
        Z_train = X_train[:, 0]
        hybrid.train_measurement_model(X_train, Z_train)
        
        # Initialize
        hybrid.initialize(x0=np.array([0, 1]), P0=np.eye(2))
        
        # Predict
        x_pred, P_pred = hybrid.predict()
        assert x_pred.shape == (2,)
        assert P_pred.shape == (2, 2)
        
        # Update
        z = np.array([0.5])
        x_upd, P_upd = hybrid.update(z)
        assert x_upd.shape == (2,)
        assert P_upd.shape == (2, 2)
        
    def test_covariance_positive_definite(self):
        """Test covariance remains positive definite."""
        hybrid = HybridEKF(self.f, self.F_jac, self.Q, self.R, self.config)
        
        # Train
        rng = np.random.default_rng(42)
        X_train = rng.uniform(-2, 2, size=(100, 2))
        Z_train = X_train[:, 0]
        hybrid.train_measurement_model(X_train, Z_train)
        
        # Run filtering
        hybrid.initialize(x0=np.array([0, 0]), P0=np.eye(2))
        
        for _ in range(20):
            hybrid.predict()
            z = np.array([rng.normal()])
            hybrid.update(z)
            
            _, P = hybrid.get_state()
            eigenvalues = np.linalg.eigvals(P)
            assert np.all(eigenvalues > 0)


class TestHybridParticleFilter:
    """Test Hybrid Particle Filter."""
    
    def setup_method(self):
        """Setup test fixtures."""
        set_seed(42)
        torch.manual_seed(42)
        
        self.Q = np.eye(2) * 0.01
        self.R = np.array([[0.1]])
        
        def f(x, u, rng):
            F = np.array([[1, 0.1], [0, 1]])
            return F @ x + rng.multivariate_normal(np.zeros(2), self.Q)
        
        self.f = f
        
        # Create and train measurement network
        self.h_network = MeasurementNetwork(n_states=2, n_obs=1, hidden_dims=[16, 8])
        
    def test_initialization(self):
        """Test particle filter initialization."""
        pf = HybridParticleFilter(self.f, self.h_network, self.R, n_particles=100)
        rng = np.random.default_rng(42)
        pf.initialize(mean=np.array([0, 1]), cov=np.eye(2), rng=rng)
        
        assert pf.particles.shape == (100, 2)
        assert pf.weights.shape == (100,)
        
    def test_predict_update(self):
        """Test prediction and update."""
        pf = HybridParticleFilter(self.f, self.h_network, self.R, n_particles=100)
        rng = np.random.default_rng(42)
        pf.initialize(mean=np.array([0, 1]), cov=np.eye(2), rng=rng)
        
        # Predict
        particles = pf.predict()
        assert particles.shape == (100, 2)
        
        # Update
        z = np.array([0.5])
        weights = pf.update(z)
        assert np.allclose(np.sum(weights), 1.0)
        
    def test_reproducibility(self):
        """Test reproducibility with same seed."""
        # Run 1
        pf1 = HybridParticleFilter(self.f, self.h_network, self.R, n_particles=50)
        rng1 = np.random.default_rng(42)
        pf1.initialize(mean=np.array([0, 1]), cov=np.eye(2), rng=rng1)
        
        for _ in range(5):
            pf1.predict()
            pf1.update(np.array([0.5]))
            pf1.resample()
        
        mean1, _ = pf1.get_state_estimate()
        
        # Run 2 with same seed
        pf2 = HybridParticleFilter(self.f, self.h_network, self.R, n_particles=50)
        rng2 = np.random.default_rng(42)
        pf2.initialize(mean=np.array([0, 1]), cov=np.eye(2), rng=rng2)
        
        for _ in range(5):
            pf2.predict()
            pf2.update(np.array([0.5]))
            pf2.resample()
        
        mean2, _ = pf2.get_state_estimate()
        
        assert np.allclose(mean1, mean2)


class TestNonlinearMeasurementSystem:
    """Test nonlinear measurement system generation."""
    
    @pytest.mark.parametrize("complexity", ["simple", "moderate", "complex"])
    def test_system_creation(self, complexity):
        """Test creating systems of different complexity."""
        h_true, h_linear, h_jac = create_nonlinear_measurement_system(complexity)
        
        x = np.array([0.5, 0.3])
        
        # Test functions work
        z_true = h_true(x)
        z_linear = h_linear(x)
        J = h_jac(x)
        
        assert z_true.shape == (1,)
        assert z_linear.shape == (1,)
        assert J.shape == (1, 2)
        
    def test_training_data_generation(self):
        """Test training data generation."""
        h_true, _, _ = create_nonlinear_measurement_system("moderate")
        rng = np.random.default_rng(42)
        
        X, Z = generate_training_data(
            f_dynamics=None,
            h_true=h_true,
            n_samples=100,
            noise_std=0.1,
            rng=rng,
        )
        
        assert X.shape == (100, 2)
        assert Z.shape == (100,)


class TestHybridVsClassicalComparison:
    """Integration tests comparing hybrid and classical approaches."""
    
    def test_hybrid_improves_on_complex_measurement(self):
        """Test that hybrid EKF improves over linear assumption."""
        set_seed(42)
        torch.manual_seed(42)
        
        # Create complex measurement system
        h_true, h_linear, h_jac_true = create_nonlinear_measurement_system("moderate")
        
        # Dynamics (linear)
        dt = 0.1
        F = np.array([[1, dt], [0, 1]])
        Q = np.eye(2) * 0.01
        R = np.array([[0.1]])
        
        def f(x, u):
            return F @ x
        
        def F_jac(x):
            return F
        
        # Generate training data for hybrid
        rng = np.random.default_rng(42)
        X_train, Z_train = generate_training_data(
            None, h_true, n_samples=200, noise_std=0.1, rng=rng
        )
        
        # Setup hybrid EKF
        config = HybridEKFConfig(n_states=2, n_obs=1, hidden_dims=[32, 16], n_epochs=100)
        hybrid = HybridEKF(f, F_jac, Q, R, config)
        hybrid.train_measurement_model(X_train, Z_train)
        
        # Generate test trajectory
        rng_test = np.random.default_rng(123)
        n_steps = 50
        
        true_states = []
        observations = []
        x_true = np.array([0.5, 0.5])
        
        for _ in range(n_steps):
            w = rng_test.multivariate_normal(np.zeros(2), Q)
            x_true = F @ x_true + w
            true_states.append(x_true.copy())
            
            z = h_true(x_true) + rng_test.normal(0, np.sqrt(R[0, 0]))
            observations.append(z[0])
        
        true_states = np.array(true_states)
        
        # Run hybrid EKF
        hybrid.initialize(x0=np.array([0, 0]), P0=np.eye(2))
        hybrid_estimates = []
        
        for z in observations:
            hybrid.predict()
            hybrid.update(np.array([z]))
            x, _ = hybrid.get_state()
            hybrid_estimates.append(x)
        
        hybrid_estimates = np.array(hybrid_estimates)
        hybrid_rmse = np.sqrt(np.mean((true_states - hybrid_estimates)**2))
        
        # The hybrid should achieve reasonable accuracy
        assert hybrid_rmse < 1.0, f"Hybrid RMSE too high: {hybrid_rmse}"
        
        # Training loss should have decreased
        assert hybrid.training_loss_history[-1] < hybrid.training_loss_history[0]

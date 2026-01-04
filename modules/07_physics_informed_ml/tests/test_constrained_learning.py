"""Tests for constrained learning with conservation laws."""

import pytest
import numpy as np
from modules._import_helper import safe_import_from

set_seed = safe_import_from('00_repo_standards.src.mlphys_core', 'set_seed')
(ConservationConfig, ConservationConstrainedNN,
 generate_pendulum_data, compute_energy_violation) = safe_import_from(
    '07_physics_informed_ml.src.constrained_learning',
    'ConservationConfig', 'ConservationConstrainedNN',
    'generate_pendulum_data', 'compute_energy_violation'
)


class TestConservationConstrainedNN:
    """Tests for conservation-constrained neural network."""
    
    def test_initialization(self):
        """Test model initializes correctly."""
        set_seed(42)
        config = ConservationConfig(epochs=10)
        model = ConservationConstrainedNN(config)
        
        assert model.model is not None
        assert model.optimizer is not None
    
    def test_training_with_energy_constraint(self):
        """Test training with energy conservation."""
        set_seed(42)
        
        # Generate data
        X_train, y_train = generate_pendulum_data(n=200, seed=42)
        
        config = ConservationConfig(
            conservation_type="energy",
            epochs=200,
            lambda_conservation=1.0,
        )
        model = ConservationConstrainedNN(config)
        
        history = model.train(X_train, y_train, verbose=0)
        
        # Loss should decrease
        assert history['loss'][-1] < history['loss'][0] * 0.5
    
    def test_prediction_shape(self):
        """Test prediction outputs correct shape."""
        set_seed(42)
        
        X_train, y_train = generate_pendulum_data(n=100, seed=42)
        
        config = ConservationConfig(epochs=10)
        model = ConservationConstrainedNN(config)
        model.train(X_train, y_train, verbose=0)
        
        X_test = np.random.randn(50, 2)
        y_pred = model.predict(X_test)
        
        assert y_pred.shape == (50, 2)
    
    def test_conservation_penalty_reduces_violation(self):
        """Test that conservation penalty reduces energy violation."""
        set_seed(42)
        
        X_train, y_train = generate_pendulum_data(n=300, seed=42)
        X_test, y_test = generate_pendulum_data(n=100, seed=99)
        
        # Model without conservation
        config_no_cons = ConservationConfig(
            conservation_type="energy",
            lambda_conservation=0.0,  # No penalty
            epochs=300,
        )
        model_no_cons = ConservationConstrainedNN(config_no_cons)
        model_no_cons.train(X_train, y_train, verbose=0)
        y_pred_no_cons = model_no_cons.predict(X_test)
        
        # Model with conservation
        config_with_cons = ConservationConfig(
            conservation_type="energy",
            lambda_conservation=1.0,  # Strong penalty
            epochs=300,
        )
        model_with_cons = ConservationConstrainedNN(config_with_cons)
        model_with_cons.train(X_train, y_train, verbose=0)
        y_pred_with_cons = model_with_cons.predict(X_test)
        
        # Compute violations
        violations_no_cons = compute_energy_violation(X_test, y_pred_no_cons)
        violations_with_cons = compute_energy_violation(X_test, y_pred_with_cons)
        
        # Conservation penalty should reduce violations
        mean_violation_no_cons = np.mean(violations_no_cons)
        mean_violation_with_cons = np.mean(violations_with_cons)
        
        assert mean_violation_with_cons < mean_violation_no_cons * 0.8  # At least 20% reduction
    
    @pytest.mark.parametrize("conservation_type", ["energy", "momentum", "mass"])
    def test_different_conservation_types(self, conservation_type):
        """Test different conservation law types."""
        set_seed(42)
        
        n_dim = 2 if conservation_type in ["energy", "momentum"] else 4
        X = np.random.randn(100, n_dim)
        y = np.random.randn(100, n_dim)
        
        config = ConservationConfig(
            input_dim=n_dim,
            output_dim=n_dim,
            conservation_type=conservation_type,
            epochs=50,
        )
        model = ConservationConstrainedNN(config)
        
        history = model.train(X, y, verbose=0)
        
        # Should complete without error
        assert len(history['loss']) > 0


class TestDataGeneration:
    """Tests for synthetic data generation."""
    
    def test_generate_pendulum_data(self):
        """Test pendulum data generation."""
        set_seed(42)
        
        n = 200
        X, y = generate_pendulum_data(n=n, omega=1.0, dt=0.1, seed=42)
        
        assert X.shape == (n, 2)
        assert y.shape == (n, 2)
        
        # Position and velocity should be reasonable
        assert np.all(np.abs(X) < 10)
        assert np.all(np.abs(y) < 10)
    
    def test_reproducible_data_generation(self):
        """Test data generation is reproducible."""
        X1, y1 = generate_pendulum_data(n=100, seed=42)
        X2, y2 = generate_pendulum_data(n=100, seed=42)
        
        assert np.array_equal(X1, X2)
        assert np.array_equal(y1, y2)
    
    def test_energy_violation_computation(self):
        """Test energy violation computation."""
        set_seed(42)
        
        X = np.array([[1.0, 0.0], [0.0, 1.0]])
        y_pred = np.array([[1.0, 0.0], [0.0, 1.0]])  # Perfect conservation
        
        violations = compute_energy_violation(X, y_pred)
        
        # Violations should be zero (or very small due to floating point)
        assert np.all(violations < 1e-10)


class TestReproducibility:
    """Tests for reproducibility."""
    
    def test_reproducible_training(self):
        """Test training is reproducible with same seed."""
        X_train, y_train = generate_pendulum_data(n=150, seed=42)
        
        config = ConservationConfig(
            conservation_type="energy",
            epochs=100,
        )
        
        # Run 1
        set_seed(42)
        model1 = ConservationConstrainedNN(config)
        history1 = model1.train(X_train, y_train, verbose=0)
        
        X_test = np.random.randn(20, 2)
        y_pred1 = model1.predict(X_test)
        
        # Run 2
        set_seed(42)
        model2 = ConservationConstrainedNN(config)
        history2 = model2.train(X_train, y_train, verbose=0)
        y_pred2 = model2.predict(X_test)
        
        # Should be identical
        assert np.allclose(history1['loss'], history2['loss'])
        assert np.allclose(y_pred1, y_pred2)

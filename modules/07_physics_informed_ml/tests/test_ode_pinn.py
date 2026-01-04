"""Tests for ODE PINN (Harmonic Oscillator)."""

import pytest
import torch
import numpy as np
from modules._import_helper import safe_import_from

set_seed = safe_import_from('00_repo_standards.src.mlphys_core', 'set_seed')
(HarmonicOscillatorConfig, HarmonicOscillatorPINN,
 solve_harmonic_oscillator_scipy, analytical_harmonic_oscillator,
 compute_energy) = safe_import_from(
    '07_physics_informed_ml.src.ode_pinn',
    'HarmonicOscillatorConfig', 'HarmonicOscillatorPINN',
    'solve_harmonic_oscillator_scipy', 'analytical_harmonic_oscillator',
    'compute_energy'
)


class TestHarmonicOscillatorPINN:
    """Tests for harmonic oscillator PINN."""
    
    def test_initialization(self):
        """Test PINN initializes correctly."""
        set_seed(42)
        config = HarmonicOscillatorConfig(omega=1.0, epochs=10)
        pinn = HarmonicOscillatorPINN(config)
        
        assert pinn.model is not None
        assert pinn.optimizer is not None
        assert pinn.t_physics.shape[0] == config.n_collocation
    
    def test_physics_residual_shape(self):
        """Test physics residual computation shape."""
        set_seed(42)
        config = HarmonicOscillatorConfig(epochs=10)
        pinn = HarmonicOscillatorPINN(config)
        
        t_test = torch.randn(20, 1, requires_grad=True)
        residual = pinn.physics_residual(pinn.model, t_test)
        
        assert residual.shape == (20, 1)
    
    def test_training_decreases_loss(self):
        """Test that training decreases loss."""
        set_seed(42)
        config = HarmonicOscillatorConfig(
            omega=1.0,
            epochs=500,
            n_collocation=50,
        )
        pinn = HarmonicOscillatorPINN(config)
        
        history = pinn.train(verbose=0)
        
        # Loss should decrease
        initial_loss = history['loss'][0]
        final_loss = history['loss'][-1]
        
        assert final_loss < initial_loss * 0.5  # At least 50% reduction
    
    def test_prediction_shape(self):
        """Test prediction outputs correct shape."""
        set_seed(42)
        config = HarmonicOscillatorConfig(epochs=10)
        pinn = HarmonicOscillatorPINN(config)
        
        t_test = np.linspace(0, 5, 50)
        x_pred = pinn.predict(t_test)
        
        assert x_pred.shape == (50,)
    
    def test_prediction_with_velocity(self):
        """Test prediction with velocity outputs correct shapes."""
        set_seed(42)
        config = HarmonicOscillatorConfig(epochs=10)
        pinn = HarmonicOscillatorPINN(config)
        
        t_test = np.linspace(0, 5, 50)
        x_pred, v_pred = pinn.predict_with_velocity(t_test)
        
        assert x_pred.shape == (50,)
        assert v_pred.shape == (50,)
    
    @pytest.mark.parametrize("omega", [1.0, 2.0, 0.5])
    def test_different_frequencies(self, omega):
        """Test PINN works with different frequencies."""
        set_seed(42)
        config = HarmonicOscillatorConfig(
            omega=omega,
            epochs=100,
            n_collocation=30,
        )
        pinn = HarmonicOscillatorPINN(config)
        
        history = pinn.train(verbose=0)
        
        # Should converge
        assert history['loss'][-1] < 1.0


class TestAnalyticalSolutions:
    """Tests for analytical and numerical solutions."""
    
    def test_analytical_solution(self):
        """Test analytical solution is correct."""
        set_seed(42)
        omega = 1.0
        x0 = 1.0
        v0 = 0.0
        t = np.linspace(0, 10, 100)
        
        x, v = analytical_harmonic_oscillator(omega, x0, v0, t)
        
        # Check initial conditions
        assert np.isclose(x[0], x0, atol=1e-6)
        assert np.isclose(v[0], v0, atol=1e-6)
        
        # Check energy conservation
        E = compute_energy(x, v, omega)
        E_var = np.std(E)
        
        assert E_var < 1e-10  # Energy should be constant
    
    def test_scipy_solver(self):
        """Test scipy solver against analytical."""
        set_seed(42)
        omega = 1.0
        x0 = 1.0
        v0 = 0.0
        t = np.linspace(0, 10, 100)
        
        x_analytical, v_analytical = analytical_harmonic_oscillator(omega, x0, v0, t)
        x_scipy, v_scipy = solve_harmonic_oscillator_scipy(omega, x0, v0, t)
        
        # Should match very closely
        x_error = np.mean((x_scipy - x_analytical) ** 2) ** 0.5
        v_error = np.mean((v_scipy - v_analytical) ** 2) ** 0.5
        
        assert x_error < 1e-6
        assert v_error < 1e-6
    
    def test_energy_computation(self):
        """Test energy computation."""
        set_seed(42)
        x = np.array([1.0, 0.0, -1.0, 0.0])
        v = np.array([0.0, 1.0, 0.0, -1.0])
        omega = 1.0
        
        E = compute_energy(x, v, omega)
        
        # For harmonic oscillator with ω=1, E = 0.5
        expected_E = 0.5 * np.ones(4)
        
        assert np.allclose(E, expected_E)


class TestReproducibility:
    """Tests for reproducibility."""
    
    def test_reproducible_training(self):
        """Test training is reproducible with same seed."""
        config = HarmonicOscillatorConfig(omega=1.0, epochs=100)
        
        # Run 1
        set_seed(42)
        pinn1 = HarmonicOscillatorPINN(config)
        history1 = pinn1.train(verbose=0)
        t_test = np.linspace(0, 5, 20)
        x1 = pinn1.predict(t_test)
        
        # Run 2
        set_seed(42)
        pinn2 = HarmonicOscillatorPINN(config)
        history2 = pinn2.train(verbose=0)
        x2 = pinn2.predict(t_test)
        
        # Should be identical
        assert np.allclose(history1['loss'], history2['loss'])
        assert np.allclose(x1, x2)

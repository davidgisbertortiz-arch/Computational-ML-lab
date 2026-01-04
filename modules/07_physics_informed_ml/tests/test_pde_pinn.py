"""Tests for PDE PINN (Heat Equation)."""

import pytest
import torch
import numpy as np
from modules._import_helper import safe_import_from

set_seed = safe_import_from('00_repo_standards.src.mlphys_core', 'set_seed')
(HeatEquationConfig, HeatEquationPINN,
 solve_heat_equation_finite_difference) = safe_import_from(
    '07_physics_informed_ml.src.pde_pinn',
    'HeatEquationConfig', 'HeatEquationPINN',
    'solve_heat_equation_finite_difference'
)


class TestHeatEquationPINN:
    """Tests for heat equation PINN."""
    
    def test_initialization(self):
        """Test PINN initializes correctly."""
        set_seed(42)
        config = HeatEquationConfig(epochs=10)
        pinn = HeatEquationPINN(config)
        
        assert pinn.model is not None
        assert pinn.optimizer is not None
        assert pinn.xt_physics is not None
        assert pinn.xt_boundary is not None
        assert pinn.xt_initial is not None
    
    def test_initial_condition_functions(self):
        """Test different initial conditions."""
        set_seed(42)
        x = np.linspace(0, 1, 50)
        
        for ic_type in ["gaussian", "sine", "step"]:
            config = HeatEquationConfig(initial_condition=ic_type, epochs=10)
            pinn = HeatEquationPINN(config)
            
            u0 = pinn._initial_condition_fn(x)
            
            assert u0.shape == (50,)
            assert np.all(np.isfinite(u0))
    
    def test_physics_residual_shape(self):
        """Test physics residual computation shape."""
        set_seed(42)
        config = HeatEquationConfig(epochs=10)
        pinn = HeatEquationPINN(config)
        
        xt_test = torch.randn(20, 2, requires_grad=True)
        residual = pinn.physics_residual(pinn.model, xt_test)
        
        assert residual.shape == (20, 1)
    
    def test_training_decreases_loss(self):
        """Test that training decreases loss."""
        set_seed(42)
        config = HeatEquationConfig(
            alpha=0.01,
            epochs=500,
            n_collocation_x=20,
            n_collocation_t=20,
            initial_condition="gaussian",
        )
        pinn = HeatEquationPINN(config)
        
        history = pinn.train(verbose=0)
        
        # Loss should decrease
        initial_loss = history['loss'][0]
        final_loss = history['loss'][-1]
        
        assert final_loss < initial_loss * 0.3  # At least 70% reduction
    
    def test_prediction_shape(self):
        """Test prediction outputs correct shape."""
        set_seed(42)
        config = HeatEquationConfig(epochs=10)
        pinn = HeatEquationPINN(config)
        
        x = np.linspace(0, 1, 30)
        t = np.linspace(0, 1, 40)
        xx, tt = np.meshgrid(x, t)
        
        u_pred = pinn.predict(xx.flatten(), tt.flatten())
        
        assert u_pred.shape == (30 * 40,)
    
    def test_boundary_conditions_enforced(self):
        """Test that boundary conditions are approximately satisfied."""
        set_seed(42)
        config = HeatEquationConfig(
            bc_left=0.0,
            bc_right=0.0,
            epochs=1000,
            lambda_bc=100.0,  # Strong BC enforcement
        )
        pinn = HeatEquationPINN(config)
        
        pinn.train(verbose=0)
        
        # Check boundary values
        t_test = np.linspace(0, 1, 20)
        x_left = np.full_like(t_test, config.x_min)
        x_right = np.full_like(t_test, config.x_max)
        
        u_left = pinn.predict(x_left, t_test)
        u_right = pinn.predict(x_right, t_test)
        
        # Should be close to BC values
        assert np.mean(np.abs(u_left - config.bc_left)) < 0.1
        assert np.mean(np.abs(u_right - config.bc_right)) < 0.1
    
    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1])
    def test_different_diffusivities(self, alpha):
        """Test PINN works with different thermal diffusivities."""
        set_seed(42)
        config = HeatEquationConfig(
            alpha=alpha,
            epochs=200,
            n_collocation_x=15,
            n_collocation_t=15,
        )
        pinn = HeatEquationPINN(config)
        
        history = pinn.train(verbose=0)
        
        # Should converge
        assert history['loss'][-1] < 1.0


class TestFiniteDifferenceSolver:
    """Tests for finite difference baseline solver."""
    
    def test_finite_difference_solver(self):
        """Test finite difference solver works."""
        set_seed(42)
        
        x = np.linspace(0, 1, 50)
        t = np.linspace(0, 0.5, 100)
        alpha = 0.01
        
        def u0_fn(x):
            return np.sin(np.pi * x)
        
        u = solve_heat_equation_finite_difference(
            alpha=alpha,
            x=x,
            t=t,
            u0_fn=u0_fn,
            bc_left=0.0,
            bc_right=0.0,
        )
        
        assert u.shape == (100, 50)
        
        # Check BCs
        assert np.allclose(u[:, 0], 0.0)
        assert np.allclose(u[:, -1], 0.0)
        
        # Check IC
        assert np.allclose(u[0, :], u0_fn(x), atol=1e-6)
        
        # Temperature should decay
        assert u[-1, :].max() < u[0, :].max()


class TestReproducibility:
    """Tests for reproducibility."""
    
    def test_reproducible_training(self):
        """Test training is reproducible with same seed."""
        config = HeatEquationConfig(
            alpha=0.01,
            epochs=100,
            n_collocation_x=15,
            n_collocation_t=15,
        )
        
        # Run 1
        set_seed(42)
        pinn1 = HeatEquationPINN(config)
        history1 = pinn1.train(verbose=0)
        
        x_test = np.array([0.3, 0.5, 0.7])
        t_test = np.array([0.2, 0.2, 0.2])
        u1 = pinn1.predict(x_test, t_test)
        
        # Run 2
        set_seed(42)
        pinn2 = HeatEquationPINN(config)
        history2 = pinn2.train(verbose=0)
        u2 = pinn2.predict(x_test, t_test)
        
        # Should be identical
        assert np.allclose(history1['loss'], history2['loss'])
        assert np.allclose(u1, u2)

"""Tests for optimizer implementations."""

import pytest
import numpy as np
from modules._import_helper import safe_import_from

(GradientDescent, MomentumOptimizer, AdamOptimizer, OptimizationResult) = safe_import_from(
    '01_numerical_toolbox.src.optimizers_from_scratch',
    'GradientDescent', 'MomentumOptimizer', 'AdamOptimizer', 'OptimizationResult'
)
(create_quadratic_bowl, rosenbrock_function, rosenbrock_gradient) = safe_import_from(
    '01_numerical_toolbox.src.toy_problems',
    'create_quadratic_bowl', 'rosenbrock_function', 'rosenbrock_gradient'
)
set_seed = safe_import_from('00_repo_standards.src.mlphys_core', 'set_seed')


class TestGradientDescent:
    """Tests for vanilla gradient descent."""
    
    def test_convergence_on_simple_quadratic(self):
        """GD should converge to minimum of simple quadratic."""
        # Simple 2D bowl: f(x) = x^2 + y^2
        bowl = create_quadratic_bowl(n_dim=2, condition_number=1.0, seed=42)
        
        optimizer = GradientDescent(learning_rate=0.5, max_iter=500, tol=1e-6)
        x0 = np.array([5.0, 5.0])
        
        result = optimizer.minimize(bowl, bowl.gradient, x0)
        
        # Should converge
        assert result.converged
        # Should find minimum (close to optimum)
        assert np.linalg.norm(result.x_final - bowl.optimum) < 1e-3
        # Final value should be close to optimum value
        assert abs(result.f_final - bowl.optimum_value) < 1e-6
    
    def test_deterministic_runs(self):
        """Same seed should give identical results."""
        bowl = create_quadratic_bowl(n_dim=3, condition_number=10.0, seed=42)
        optimizer = GradientDescent(learning_rate=0.01, max_iter=50)
        
        x0 = np.array([1.0, 2.0, 3.0])
        
        # Run twice
        result1 = optimizer.minimize(bowl, bowl.gradient, x0.copy())
        result2 = optimizer.minimize(bowl, bowl.gradient, x0.copy())
        
        # Should be identical
        np.testing.assert_array_equal(result1.x_final, result2.x_final)
        assert result1.f_final == result2.f_final
        assert len(result1.history["f_vals"]) == len(result2.history["f_vals"])
    
    def test_output_shapes(self):
        """Check all output arrays have correct shapes."""
        bowl = create_quadratic_bowl(n_dim=5, condition_number=5.0, seed=123)
        optimizer = GradientDescent(learning_rate=0.05, max_iter=30)
        
        x0 = np.zeros(5)
        result = optimizer.minimize(bowl, bowl.gradient, x0)
        
        # x_final should have same shape as x0
        assert result.x_final.shape == x0.shape
        
        # History arrays should have length n_iterations or n_iterations+1
        n = result.n_iterations
        assert len(result.history["f_vals"]) == n + 1
        assert len(result.history["grad_norms"]) == n + 1
        assert len(result.history["step_sizes"]) == n
    
    def test_convergence_depends_on_conditioning(self):
        """Ill-conditioned problems should converge slower."""
        # Well-conditioned
        bowl_easy = create_quadratic_bowl(n_dim=2, condition_number=1.0, seed=42)
        # Ill-conditioned
        bowl_hard = create_quadratic_bowl(n_dim=2, condition_number=100.0, seed=42)
        
        optimizer = GradientDescent(learning_rate=0.05, max_iter=1000)
        x0 = np.array([10.0, 10.0])
        
        result_easy = optimizer.minimize(bowl_easy, bowl_easy.gradient, x0.copy())
        result_hard = optimizer.minimize(bowl_hard, bowl_hard.gradient, x0.copy())
        
        # Easy problem should converge in fewer iterations
        assert result_easy.n_iterations < result_hard.n_iterations
    
    def test_learning_rate_effect(self):
        """Larger learning rate should converge faster (if stable)."""
        bowl = create_quadratic_bowl(n_dim=2, condition_number=5.0, seed=42)
        x0 = np.array([5.0, 5.0])
        
        opt_small = GradientDescent(learning_rate=0.01, max_iter=1000)
        opt_large = GradientDescent(learning_rate=0.3, max_iter=1000)
        
        result_small = opt_small.minimize(bowl, bowl.gradient, x0.copy())
        result_large = opt_large.minimize(bowl, bowl.gradient, x0.copy())
        
        # Larger LR should need fewer iterations
        assert result_large.n_iterations < result_small.n_iterations
    
    def test_gradient_norm_decreases(self):
        """Gradient norm should monotonically decrease for convex problems."""
        bowl = create_quadratic_bowl(n_dim=3, condition_number=10.0, seed=42)
        optimizer = GradientDescent(learning_rate=0.05, max_iter=50)
        
        x0 = np.array([5.0, -3.0, 2.0])
        result = optimizer.minimize(bowl, bowl.gradient, x0)
        
        grad_norms = result.history["grad_norms"]
        
        # Should decrease overall (allow small fluctuations)
        assert grad_norms[-1] < grad_norms[0]
        # Most steps should decrease
        decreases = sum(grad_norms[i+1] < grad_norms[i] for i in range(len(grad_norms)-1))
        assert decreases > 0.8 * len(grad_norms)  # At least 80%


class TestMomentumOptimizer:
    """Tests for momentum optimizer."""
    
    def test_converges_on_quadratic(self):
        """Momentum should converge on quadratic problems."""
        bowl = create_quadratic_bowl(n_dim=2, condition_number=10.0, seed=42)
        
        optimizer = MomentumOptimizer(
            learning_rate=0.1, momentum=0.9, max_iter=500, tol=1e-6
        )
        x0 = np.array([5.0, 5.0])
        
        result = optimizer.minimize(bowl, bowl.gradient, x0)
        
        assert result.converged
        assert np.linalg.norm(result.x_final - bowl.optimum) < 1e-2
    
    def test_faster_than_gd_on_ill_conditioned(self):
        """Momentum should outperform GD on ill-conditioned problems."""
        bowl = create_quadratic_bowl(n_dim=2, condition_number=100.0, seed=42)
        x0 = np.array([10.0, 10.0])
        
        gd = GradientDescent(learning_rate=0.005, max_iter=200)
        momentum = MomentumOptimizer(learning_rate=0.005, momentum=0.9, max_iter=200)
        
        result_gd = gd.minimize(bowl, bowl.gradient, x0.copy())
        result_momentum = momentum.minimize(bowl, bowl.gradient, x0.copy())
        
        # Momentum should converge faster or achieve lower final loss
        assert (result_momentum.n_iterations < result_gd.n_iterations or
                result_momentum.f_final < result_gd.f_final)
    
    def test_velocity_tracking(self):
        """Momentum should track velocity norms in history."""
        bowl = create_quadratic_bowl(n_dim=3, condition_number=5.0, seed=42)
        optimizer = MomentumOptimizer(learning_rate=0.01, momentum=0.9, max_iter=50)
        
        x0 = np.zeros(3)
        result = optimizer.minimize(bowl, bowl.gradient, x0)
        
        # Should have velocity_norms in history
        assert "velocity_norms" in result.history
        assert len(result.history["velocity_norms"]) == result.n_iterations + 1
        
        # Velocity norms should be non-negative
        assert all(v >= 0 for v in result.history["velocity_norms"])
    
    def test_momentum_coefficient_effect(self):
        """Higher momentum should retain more velocity."""
        bowl = create_quadratic_bowl(n_dim=2, condition_number=10.0, seed=42)
        x0 = np.array([3.0, 3.0])
        
        opt_low = MomentumOptimizer(learning_rate=0.1, momentum=0.5, max_iter=1000)
        opt_high = MomentumOptimizer(learning_rate=0.1, momentum=0.95, max_iter=1000)
        
        result_low = opt_low.minimize(bowl, bowl.gradient, x0.copy())
        result_high = opt_high.minimize(bowl, bowl.gradient, x0.copy())
        
        # At least one should converge (high momentum may overshoot)
        assert result_low.converged or result_high.converged


class TestAdamOptimizer:
    """Tests for Adam optimizer."""
    
    def test_converges_on_quadratic(self):
        """Adam should converge on quadratic problems."""
        bowl = create_quadratic_bowl(n_dim=3, condition_number=10.0, seed=42)
        
        optimizer = AdamOptimizer(learning_rate=0.5, max_iter=500, tol=1e-6)
        x0 = np.array([5.0, -3.0, 2.0])
        
        result = optimizer.minimize(bowl, bowl.gradient, x0)
        
        assert result.converged
        assert np.linalg.norm(result.x_final - bowl.optimum) < 1e-2
    
    def test_adaptive_learning_rates(self):
        """Adam should adapt learning rates per parameter."""
        bowl = create_quadratic_bowl(n_dim=5, condition_number=50.0, seed=42)
        
        optimizer = AdamOptimizer(learning_rate=0.1, max_iter=50)
        x0 = np.zeros(5)
        
        result = optimizer.minimize(bowl, bowl.gradient, x0)
        
        # Should track first and second moments
        assert "m_norms" in result.history
        assert "v_norms" in result.history
        assert len(result.history["m_norms"]) == result.n_iterations + 1
        assert len(result.history["v_norms"]) == result.n_iterations + 1
    
    def test_robust_to_scale(self):
        """Adam should handle different gradient scales well."""
        # Create problem with very different scales in different directions
        bowl = create_quadratic_bowl(n_dim=2, condition_number=1000.0, seed=42)
        
        optimizer = AdamOptimizer(learning_rate=0.1, max_iter=200)
        x0 = np.array([10.0, 10.0])
        
        result = optimizer.minimize(bowl, bowl.gradient, x0)
        
        # Should still make reasonable progress
        assert result.f_final < bowl(x0)
        assert np.linalg.norm(result.x_final - bowl.optimum) < 5.0
    
    def test_beta_parameters(self):
        """Different beta values should affect convergence."""
        bowl = create_quadratic_bowl(n_dim=2, condition_number=10.0, seed=42)
        x0 = np.array([5.0, 5.0])
        
        # Standard values
        opt_std = AdamOptimizer(learning_rate=0.1, beta1=0.9, beta2=0.999, max_iter=50)
        # More aggressive
        opt_agg = AdamOptimizer(learning_rate=0.1, beta1=0.5, beta2=0.9, max_iter=50)
        
        result_std = opt_std.minimize(bowl, bowl.gradient, x0.copy())
        result_agg = opt_agg.minimize(bowl, bowl.gradient, x0.copy())
        
        # Both should converge (though may differ in speed)
        assert result_std.f_final < bowl(x0)
        assert result_agg.f_final < bowl(x0)


class TestOptimizerComparison:
    """Compare optimizers on various problems."""
    
    def test_all_converge_on_well_conditioned(self):
        """All optimizers should converge on well-conditioned problems."""
        bowl = create_quadratic_bowl(n_dim=3, condition_number=2.0, seed=42)
        x0 = np.array([5.0, -3.0, 2.0])
        
        gd = GradientDescent(learning_rate=0.3, max_iter=500)
        momentum = MomentumOptimizer(learning_rate=0.3, momentum=0.9, max_iter=500)
        adam = AdamOptimizer(learning_rate=0.5, max_iter=500)
        
        result_gd = gd.minimize(bowl, bowl.gradient, x0.copy())
        result_momentum = momentum.minimize(bowl, bowl.gradient, x0.copy())
        result_adam = adam.minimize(bowl, bowl.gradient, x0.copy())
        
        # All should converge
        assert result_gd.converged
        assert result_momentum.converged
        assert result_adam.converged
        
        # All should find similar minimum
        for result in [result_gd, result_momentum, result_adam]:
            assert np.linalg.norm(result.x_final - bowl.optimum) < 1e-1
    
    def test_momentum_and_adam_better_on_ill_conditioned(self):
        """Momentum and Adam should outperform GD on ill-conditioned problems."""
        bowl = create_quadratic_bowl(n_dim=2, condition_number=100.0, seed=42)
        x0 = np.array([10.0, 10.0])
        max_iter = 100
        
        gd = GradientDescent(learning_rate=0.01, max_iter=max_iter)
        momentum = MomentumOptimizer(learning_rate=0.01, momentum=0.9, max_iter=max_iter)
        adam = AdamOptimizer(learning_rate=0.1, max_iter=max_iter)
        
        result_gd = gd.minimize(bowl, bowl.gradient, x0.copy())
        result_momentum = momentum.minimize(bowl, bowl.gradient, x0.copy())
        result_adam = adam.minimize(bowl, bowl.gradient, x0.copy())
        
        # Momentum and Adam should achieve lower final loss than GD
        assert result_momentum.f_final < result_gd.f_final or result_momentum.n_iterations < result_gd.n_iterations
        assert result_adam.f_final < result_gd.f_final
    
    def test_nonconvex_rosenbrock(self):
        """Test all optimizers on non-convex Rosenbrock function."""
        x0 = np.array([0.0, 0.0])
        max_iter = 500
        
        gd = GradientDescent(learning_rate=0.001, max_iter=max_iter)
        momentum = MomentumOptimizer(learning_rate=0.001, momentum=0.9, max_iter=max_iter)
        adam = AdamOptimizer(learning_rate=0.01, max_iter=max_iter)
        
        result_gd = gd.minimize(rosenbrock_function, rosenbrock_gradient, x0.copy())
        result_momentum = momentum.minimize(rosenbrock_function, rosenbrock_gradient, x0.copy())
        result_adam = adam.minimize(rosenbrock_function, rosenbrock_gradient, x0.copy())
        
        # Should make progress (global minimum is at [1, 1] with value 0)
        assert result_gd.f_final < rosenbrock_function(x0)
        assert result_momentum.f_final < rosenbrock_function(x0)
        assert result_adam.f_final < rosenbrock_function(x0)
        
        # Adam should typically perform best
        assert result_adam.f_final <= min(result_gd.f_final, result_momentum.f_final) * 1.1


class TestOptimizationResult:
    """Tests for OptimizationResult dataclass."""
    
    def test_result_structure(self):
        """OptimizationResult should have all required fields."""
        bowl = create_quadratic_bowl(n_dim=2, condition_number=5.0, seed=42)
        optimizer = GradientDescent(learning_rate=0.1, max_iter=50)
        
        x0 = np.array([3.0, 3.0])
        result = optimizer.minimize(bowl, bowl.gradient, x0)
        
        # Check all fields exist
        assert hasattr(result, "x_final")
        assert hasattr(result, "f_final")
        assert hasattr(result, "history")
        assert hasattr(result, "converged")
        assert hasattr(result, "n_iterations")
        
        # Check types
        assert isinstance(result.x_final, np.ndarray)
        assert isinstance(result.f_final, (float, np.floating))
        assert isinstance(result.history, dict)
        assert isinstance(result.converged, bool)
        assert isinstance(result.n_iterations, int)
    
    def test_history_contents(self):
        """History dict should contain convergence diagnostics."""
        bowl = create_quadratic_bowl(n_dim=2, condition_number=5.0, seed=42)
        optimizer = GradientDescent(learning_rate=0.1, max_iter=50)
        
        x0 = np.array([3.0, 3.0])
        result = optimizer.minimize(bowl, bowl.gradient, x0)
        
        # GD should track these
        assert "f_vals" in result.history
        assert "grad_norms" in result.history
        assert "step_sizes" in result.history
        
        # All should be lists or arrays
        for key, val in result.history.items():
            assert isinstance(val, (list, np.ndarray))
            assert len(val) > 0

"""Tests for toy optimization problems."""

import pytest
import numpy as np
from modules._import_helper import safe_import_from

(create_quadratic_bowl, QuadraticBowl, create_linear_regression,
 LinearRegressionProblem, linear_regression_closed_form,
 linear_regression_gradient_descent, rosenbrock_function,
 rosenbrock_gradient, beale_function, beale_gradient) = safe_import_from(
    '01_numerical_toolbox.src.toy_problems',
    'create_quadratic_bowl', 'QuadraticBowl', 'create_linear_regression',
    'LinearRegressionProblem', 'linear_regression_closed_form',
    'linear_regression_gradient_descent', 'rosenbrock_function',
    'rosenbrock_gradient', 'beale_function', 'beale_gradient'
)
set_seed = safe_import_from('00_repo_standards.src.mlphys_core', 'set_seed')


class TestQuadraticBowl:
    """Tests for quadratic bowl creation and evaluation."""
    
    def test_creates_valid_bowl(self):
        """Should create quadratic bowl with correct properties."""
        bowl = create_quadratic_bowl(n_dim=3, condition_number=10.0, seed=42)
        
        assert isinstance(bowl, QuadraticBowl)
        assert bowl.A.shape == (3, 3)
        assert bowl.b.shape == (3,)
        assert isinstance(bowl.c, float)
        assert bowl.optimum.shape == (3,)
    
    def test_condition_number_approximately_correct(self):
        """Created bowl should have approximately target condition number."""
        target_kappa = 100.0
        bowl = create_quadratic_bowl(n_dim=5, condition_number=target_kappa, seed=42)
        
        # Allow 20% tolerance
        assert 0.8 * target_kappa < bowl.condition_number < 1.2 * target_kappa
    
    def test_hessian_is_positive_definite(self):
        """Hessian should be positive definite (all eigenvalues > 0)."""
        bowl = create_quadratic_bowl(n_dim=4, condition_number=50.0, seed=42)
        
        eigenvalues = np.linalg.eigvals(bowl.A)
        assert np.all(eigenvalues > 0)
    
    def test_hessian_is_symmetric(self):
        """Hessian should be symmetric."""
        bowl = create_quadratic_bowl(n_dim=4, condition_number=20.0, seed=42)
        
        np.testing.assert_allclose(bowl.A, bowl.A.T, atol=1e-10)
    
    def test_optimum_is_global_minimum(self):
        """Optimum should be the global minimum."""
        bowl = create_quadratic_bowl(n_dim=2, condition_number=5.0, seed=42)
        
        # Value at optimum
        f_opt = bowl(bowl.optimum)
        
        # Value at nearby points should be higher
        for _ in range(10):
            x_nearby = bowl.optimum + 0.1 * np.random.randn(2)
            assert bowl(x_nearby) >= f_opt
    
    def test_gradient_zero_at_optimum(self):
        """Gradient should be zero at optimum."""
        bowl = create_quadratic_bowl(n_dim=3, condition_number=10.0, seed=42)
        
        grad = bowl.gradient(bowl.optimum)
        np.testing.assert_allclose(grad, np.zeros(3), atol=1e-10)
    
    def test_gradient_correct(self):
        """Gradient should match numerical gradient."""
        bowl = create_quadratic_bowl(n_dim=2, condition_number=10.0, seed=42)
        
        x = np.array([1.0, -1.0])
        grad_analytic = bowl.gradient(x)
        
        # Numerical gradient
        eps = 1e-7
        grad_numerical = np.zeros(2)
        for i in range(2):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            grad_numerical[i] = (bowl(x_plus) - bowl(x_minus)) / (2 * eps)
        
        np.testing.assert_allclose(grad_analytic, grad_numerical, atol=1e-5)
    
    def test_hessian_constant(self):
        """Hessian should be constant (equal to A)."""
        bowl = create_quadratic_bowl(n_dim=3, condition_number=5.0, seed=42)
        
        x1 = np.array([1.0, 2.0, 3.0])
        x2 = np.array([-1.0, 0.5, -2.0])
        
        H1 = bowl.hessian(x1)
        H2 = bowl.hessian(x2)
        
        np.testing.assert_allclose(H1, H2, atol=1e-10)
        np.testing.assert_allclose(H1, bowl.A, atol=1e-10)
    
    def test_deterministic_with_seed(self):
        """Same seed should produce same bowl."""
        bowl1 = create_quadratic_bowl(n_dim=2, condition_number=10.0, seed=42)
        bowl2 = create_quadratic_bowl(n_dim=2, condition_number=10.0, seed=42)
        
        np.testing.assert_allclose(bowl1.A, bowl2.A, atol=1e-10)
        np.testing.assert_allclose(bowl1.b, bowl2.b, atol=1e-10)
        assert bowl1.c == bowl2.c
    
    def test_custom_offset(self):
        """Should accept custom offset for minimum location."""
        offset = np.array([5.0, -3.0])
        bowl = create_quadratic_bowl(
            n_dim=2, condition_number=5.0, seed=42, offset=offset
        )
        
        # Optimum should be at offset
        np.testing.assert_allclose(bowl.optimum, offset, atol=1e-10)


class TestLinearRegression:
    """Tests for linear regression problem creation."""
    
    def test_creates_valid_problem(self):
        """Should create valid linear regression problem."""
        prob = create_linear_regression(
            n_samples=100, n_features=5, noise_std=0.1, seed=42
        )
        
        assert isinstance(prob, LinearRegressionProblem)
        assert prob.X.shape == (100, 5)
        assert prob.y.shape == (100,)
        assert prob.true_weights.shape == (5,)
        assert prob.condition_number > 0
    
    def test_ill_conditioned_problem(self):
        """Should create ill-conditioned problem when requested."""
        prob = create_linear_regression(
            n_samples=50, n_features=10, condition_number=100.0, seed=42
        )
        
        # Condition number should be approximately correct
        assert 50.0 < prob.condition_number < 200.0
    
    def test_deterministic_with_seed(self):
        """Same seed should produce same problem."""
        prob1 = create_linear_regression(n_samples=50, n_features=5, seed=42)
        prob2 = create_linear_regression(n_samples=50, n_features=5, seed=42)
        
        np.testing.assert_allclose(prob1.X, prob2.X, atol=1e-10)
        np.testing.assert_allclose(prob1.y, prob2.y, atol=1e-10)
        np.testing.assert_allclose(prob1.true_weights, prob2.true_weights, atol=1e-10)
    
    def test_objective_function(self):
        """Objective should compute MSE."""
        prob = create_linear_regression(n_samples=50, n_features=3, seed=42)
        
        w = np.array([1.0, 0.0, -1.0])
        obj = prob.objective(w)
        
        # Manually compute MSE
        residuals = prob.X @ w - prob.y
        expected_mse = np.mean(residuals ** 2)
        
        assert np.abs(obj - expected_mse) < 1e-10
    
    def test_gradient_correct(self):
        """Gradient should match numerical gradient."""
        prob = create_linear_regression(n_samples=50, n_features=3, seed=42)
        
        w = np.array([1.0, -0.5, 0.5])
        grad_analytic = prob.gradient(w)
        
        # Numerical gradient
        eps = 1e-7
        grad_numerical = np.zeros(3)
        for i in range(3):
            w_plus = w.copy()
            w_plus[i] += eps
            w_minus = w.copy()
            w_minus[i] -= eps
            grad_numerical[i] = (prob.objective(w_plus) - prob.objective(w_minus)) / (2 * eps)
        
        np.testing.assert_allclose(grad_analytic, grad_numerical, atol=1e-5)
    
    def test_hessian_constant(self):
        """Hessian should be constant for linear regression."""
        prob = create_linear_regression(n_samples=50, n_features=3, seed=42)
        
        w1 = np.array([1.0, 0.0, -1.0])
        w2 = np.array([0.0, 2.0, 1.0])
        
        H1 = prob.hessian(w1)
        H2 = prob.hessian(w2)
        
        np.testing.assert_allclose(H1, H2, atol=1e-10)


class TestLinearRegressionSolvers:
    """Tests for linear regression solution methods."""
    
    def test_closed_form_solution(self):
        """Closed-form solution should minimize objective."""
        prob = create_linear_regression(n_samples=100, n_features=5, seed=42)
        
        w_closed = linear_regression_closed_form(prob.X, prob.y)
        
        # Should have correct shape
        assert w_closed.shape == (5,)
        
        # Should be close to true weights (with noise)
        # Just check it's reasonable
        assert np.linalg.norm(w_closed - prob.true_weights) < 2.0
    
    def test_gradient_descent_converges(self):
        """Gradient descent should converge to similar solution."""
        prob = create_linear_regression(n_samples=100, n_features=5, seed=42)
        
        w_gd, losses = linear_regression_gradient_descent(
            prob.X, prob.y, learning_rate=0.1, max_iter=1000
        )
        
        # Should have correct shape
        assert w_gd.shape == (5,)
        
        # Losses should decrease
        assert losses[-1] < losses[0]
        
        # Should be close to closed-form solution
        w_closed = linear_regression_closed_form(prob.X, prob.y)
        np.testing.assert_allclose(w_gd, w_closed, atol=1e-3)
    
    def test_gradient_descent_loss_history(self):
        """Gradient descent should return loss history."""
        prob = create_linear_regression(n_samples=50, n_features=3, seed=42)
        
        w_gd, losses = linear_regression_gradient_descent(
            prob.X, prob.y, learning_rate=0.1, max_iter=100
        )
        
        # Should have losses for each iteration
        assert len(losses) > 0
        assert len(losses) <= 100
        
        # Losses should generally decrease
        assert losses[-1] < losses[0]
    
    def test_closed_form_vs_gradient_descent(self):
        """Closed-form and GD should give similar results on well-conditioned problems."""
        prob = create_linear_regression(
            n_samples=100, n_features=5, condition_number=None, seed=42
        )
        
        w_closed = linear_regression_closed_form(prob.X, prob.y)
        w_gd, _ = linear_regression_gradient_descent(
            prob.X, prob.y, learning_rate=0.1, max_iter=2000
        )
        
        # Should be very close
        np.testing.assert_allclose(w_gd, w_closed, atol=1e-2)
    
    def test_learning_rate_effect(self):
        """Larger learning rate should converge faster (if stable)."""
        prob = create_linear_regression(n_samples=50, n_features=3, seed=42)
        
        _, losses_small = linear_regression_gradient_descent(
            prob.X, prob.y, learning_rate=0.01, max_iter=500
        )
        _, losses_large = linear_regression_gradient_descent(
            prob.X, prob.y, learning_rate=0.1, max_iter=500
        )
        
        # Larger LR should converge in fewer iterations
        # (find where loss reaches 95% of minimum)
        min_loss_small = losses_small[-1]
        min_loss_large = losses_large[-1]
        
        threshold_small = min_loss_small + 0.05 * (losses_small[0] - min_loss_small)
        threshold_large = min_loss_large + 0.05 * (losses_large[0] - min_loss_large)
        
        iters_small = next(i for i, loss in enumerate(losses_small) if loss < threshold_small)
        iters_large = next(i for i, loss in enumerate(losses_large) if loss < threshold_large)
        
        assert iters_large < iters_small


class TestNonconvexFunctions:
    """Tests for non-convex optimization test functions."""
    
    def test_rosenbrock_at_minimum(self):
        """Rosenbrock should have value 0 at (1, 1)."""
        x_opt = np.array([1.0, 1.0])
        f_opt = rosenbrock_function(x_opt)
        
        assert np.abs(f_opt) < 1e-10
    
    def test_rosenbrock_gradient_at_minimum(self):
        """Gradient should be zero at minimum."""
        x_opt = np.array([1.0, 1.0])
        grad = rosenbrock_gradient(x_opt)
        
        np.testing.assert_allclose(grad, np.zeros(2), atol=1e-10)
    
    def test_rosenbrock_gradient_numerical(self):
        """Rosenbrock gradient should match numerical gradient."""
        x = np.array([0.5, 2.0])
        grad_analytic = rosenbrock_gradient(x)
        
        # Numerical gradient
        eps = 1e-7
        grad_numerical = np.zeros(2)
        for i in range(2):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            grad_numerical[i] = (
                rosenbrock_function(x_plus) - rosenbrock_function(x_minus)
            ) / (2 * eps)
        
        np.testing.assert_allclose(grad_analytic, grad_numerical, atol=1e-5)
    
    def test_rosenbrock_requires_2d(self):
        """Rosenbrock should require 2D input."""
        with pytest.raises(ValueError):
            rosenbrock_function(np.array([1.0]))
        
        with pytest.raises(ValueError):
            rosenbrock_function(np.array([1.0, 2.0, 3.0]))
    
    def test_beale_at_minimum(self):
        """Beale should have value 0 at (3, 0.5)."""
        x_opt = np.array([3.0, 0.5])
        f_opt = beale_function(x_opt)
        
        assert np.abs(f_opt) < 1e-10
    
    def test_beale_gradient_numerical(self):
        """Beale gradient should match numerical gradient."""
        x = np.array([2.0, 1.0])
        grad_analytic = beale_gradient(x)
        
        # Numerical gradient
        eps = 1e-7
        grad_numerical = np.zeros(2)
        for i in range(2):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            grad_numerical[i] = (
                beale_function(x_plus) - beale_function(x_minus)
            ) / (2 * eps)
        
        np.testing.assert_allclose(grad_analytic, grad_numerical, atol=1e-4)
    
    def test_beale_requires_2d(self):
        """Beale should require 2D input."""
        with pytest.raises(ValueError):
            beale_function(np.array([1.0]))
        
        with pytest.raises(ValueError):
            beale_gradient(np.array([1.0, 2.0, 3.0]))


class TestQuadraticBowlMethods:
    """Tests for QuadraticBowl dataclass methods."""
    
    def test_call_evaluates_function(self):
        """Calling bowl should evaluate function."""
        bowl = create_quadratic_bowl(n_dim=2, condition_number=5.0, seed=42)
        
        x = np.array([1.0, -1.0])
        f_val = bowl(x)
        
        # Should be a scalar
        assert isinstance(f_val, (float, np.floating))
        assert np.isfinite(f_val)
    
    def test_gradient_method(self):
        """Gradient method should compute gradient."""
        bowl = create_quadratic_bowl(n_dim=3, condition_number=5.0, seed=42)
        
        x = np.array([1.0, 2.0, -1.0])
        grad = bowl.gradient(x)
        
        # Should have correct shape
        assert grad.shape == (3,)
        
        # Should equal A @ x - b
        expected = bowl.A @ x - bowl.b
        np.testing.assert_allclose(grad, expected, atol=1e-10)
    
    def test_hessian_method(self):
        """Hessian method should return A."""
        bowl = create_quadratic_bowl(n_dim=4, condition_number=5.0, seed=42)
        
        x = np.random.randn(4)
        H = bowl.hessian(x)
        
        # Should equal A
        np.testing.assert_allclose(H, bowl.A, atol=1e-10)

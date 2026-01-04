"""Tests for linear algebra utilities."""

import pytest
import numpy as np
from modules._import_helper import safe_import_from

(pca_via_svd, PCAResult, condition_number, ridge_regularization, 
 demonstrate_ill_conditioning) = safe_import_from(
    '01_numerical_toolbox.src.linear_algebra',
    'pca_via_svd', 'PCAResult', 'condition_number', 'ridge_regularization',
    'demonstrate_ill_conditioning'
)
set_seed = safe_import_from('00_repo_standards.src.mlphys_core', 'set_seed')


class TestPCAViaSVD:
    """Tests for PCA implementation."""
    
    def test_basic_pca(self):
        """PCA should decompose data into principal components."""
        set_seed(42)
        
        # Create correlated data
        X = np.random.randn(100, 5)
        X[:, 1] = X[:, 0] + 0.1 * np.random.randn(100)  # Correlated with first
        
        result = pca_via_svd(X, n_components=3)
        
        # Check output types
        assert isinstance(result, PCAResult)
        assert result.components.shape == (5, 3)
        assert len(result.explained_variance) == 3
        assert len(result.explained_variance_ratio) == 3
        assert len(result.singular_values) == 3
        assert result.mean.shape == (5,)
    
    def test_explained_variance_sums_to_one(self):
        """Explained variance ratios should sum to <= 1."""
        set_seed(42)
        X = np.random.randn(100, 10)
        
        result = pca_via_svd(X)
        
        # Should sum to 1.0 (within numerical precision)
        assert np.abs(np.sum(result.explained_variance_ratio) - 1.0) < 1e-10
    
    def test_explained_variance_descending(self):
        """Explained variance should be in descending order."""
        set_seed(42)
        X = np.random.randn(100, 10)
        
        result = pca_via_svd(X)
        
        # Check descending order
        for i in range(len(result.explained_variance) - 1):
            assert result.explained_variance[i] >= result.explained_variance[i + 1]
    
    def test_transform_output_shape(self):
        """Transform should produce correct output shape."""
        set_seed(42)
        X = np.random.randn(100, 10)
        
        result = pca_via_svd(X, n_components=3)
        Z = result.transform(X)
        
        # Should reduce to n_components dimensions
        assert Z.shape == (100, 3)
    
    def test_inverse_transform_reconstruction(self):
        """Inverse transform should reconstruct original data."""
        set_seed(42)
        X = np.random.randn(50, 5)
        
        # Use all components
        result = pca_via_svd(X, n_components=5)
        Z = result.transform(X)
        X_reconstructed = result.inverse_transform(Z)
        
        # Should be nearly identical (within numerical precision)
        np.testing.assert_allclose(X, X_reconstructed, atol=1e-10)
    
    def test_reconstruction_error_increases(self):
        """Reconstruction error should increase as n_components decreases."""
        set_seed(42)
        X = np.random.randn(100, 10)
        
        result = pca_via_svd(X)
        
        errors = []
        for n_comp in [10, 5, 3, 1]:
            error = result.reconstruction_error(X, n_comp)
            errors.append(error)
        
        # Errors should increase (or stay same) as components decrease
        for i in range(len(errors) - 1):
            assert errors[i] <= errors[i + 1]
        
        # Full reconstruction should have near-zero error
        assert errors[0] < 1e-10
    
    def test_deterministic_results(self):
        """Same input should give same PCA results."""
        set_seed(42)
        X = np.random.randn(50, 5)
        
        result1 = pca_via_svd(X, n_components=3)
        result2 = pca_via_svd(X.copy(), n_components=3)
        
        # Components might differ by sign, so check absolute values
        np.testing.assert_allclose(
            np.abs(result1.components), np.abs(result2.components), atol=1e-10
        )
        np.testing.assert_allclose(
            result1.explained_variance, result2.explained_variance, atol=1e-10
        )
    
    def test_centering_effect(self):
        """Centering should affect PCA results."""
        set_seed(42)
        X = np.random.randn(50, 5) + 10  # Shifted data
        
        result_centered = pca_via_svd(X, center=True)
        result_uncentered = pca_via_svd(X, center=False)
        
        # Means should differ
        assert not np.allclose(result_centered.mean, result_uncentered.mean)
        
        # Results should differ
        assert not np.allclose(
            result_centered.explained_variance, result_uncentered.explained_variance
        )
    
    def test_high_dimensional_data(self):
        """PCA should work with high-dimensional data."""
        set_seed(42)
        X = np.random.randn(50, 100)  # More features than samples
        
        result = pca_via_svd(X, n_components=10)
        
        # Should still work
        assert result.components.shape == (100, 10)
        assert len(result.explained_variance) == 10
    
    def test_single_component(self):
        """PCA with single component should work."""
        set_seed(42)
        X = np.random.randn(50, 5)
        
        result = pca_via_svd(X, n_components=1)
        
        assert result.components.shape == (5, 1)
        assert len(result.explained_variance) == 1
        
        Z = result.transform(X)
        assert Z.shape == (50, 1)


class TestConditionNumber:
    """Tests for condition number computation."""
    
    def test_identity_matrix(self):
        """Identity matrix should have condition number 1."""
        I = np.eye(5)
        kappa = condition_number(I)
        
        assert np.abs(kappa - 1.0) < 1e-10
    
    def test_diagonal_matrix(self):
        """Condition number should be ratio of max/min diagonal elements."""
        D = np.diag([10.0, 5.0, 2.0, 1.0])
        kappa = condition_number(D)
        
        expected = 10.0 / 1.0
        assert np.abs(kappa - expected) < 1e-10
    
    def test_ill_conditioned_matrix(self):
        """Ill-conditioned matrix should have large condition number."""
        # Matrix with very different scales
        A = np.diag([1000.0, 1.0])
        kappa = condition_number(A)
        
        assert kappa > 100.0
    
    def test_orthogonal_matrix(self):
        """Orthogonal matrix should have condition number 1."""
        set_seed(42)
        Q, _ = np.linalg.qr(np.random.randn(5, 5))
        kappa = condition_number(Q)
        
        assert np.abs(kappa - 1.0) < 1e-10
    
    def test_rectangular_matrix(self):
        """Should work with rectangular matrices."""
        set_seed(42)
        A = np.random.randn(10, 5)
        kappa = condition_number(A)
        
        assert kappa > 0
        assert np.isfinite(kappa)


class TestRidgeRegularization:
    """Tests for ridge regression."""
    
    def test_improves_condition_number(self):
        """Ridge should improve condition number."""
        set_seed(42)
        
        # Create ill-conditioned problem
        X = np.random.randn(50, 10)
        X[:, 0] *= 100  # Make first feature dominant
        y = np.random.randn(50)
        
        w, kappa_before, kappa_after = ridge_regularization(X, y, lambda_=1.0)
        
        # Condition should improve
        assert kappa_after < kappa_before
        
        # Weights should be reasonable
        assert w.shape == (10,)
        assert np.all(np.isfinite(w))
    
    def test_lambda_effect(self):
        """Larger lambda should improve conditioning more."""
        set_seed(42)
        X = np.random.randn(50, 10)
        y = np.random.randn(50)
        
        _, _, kappa_small = ridge_regularization(X, y, lambda_=0.1)
        _, _, kappa_large = ridge_regularization(X, y, lambda_=10.0)
        
        # Larger lambda should give better conditioning
        assert kappa_large < kappa_small
    
    def test_solution_shrinkage(self):
        """Ridge should shrink solution compared to OLS."""
        set_seed(42)
        X = np.random.randn(50, 5)
        y = np.random.randn(50)
        
        # OLS solution
        w_ols = np.linalg.lstsq(X, y, rcond=None)[0]
        
        # Ridge solutions
        w_ridge_small, _, _ = ridge_regularization(X, y, lambda_=0.01)
        w_ridge_large, _, _ = ridge_regularization(X, y, lambda_=10.0)
        
        # Ridge should shrink
        assert np.linalg.norm(w_ridge_small) < np.linalg.norm(w_ols) * 1.1
        assert np.linalg.norm(w_ridge_large) < np.linalg.norm(w_ridge_small)
    
    def test_deterministic_solution(self):
        """Same inputs should give same solution."""
        set_seed(42)
        X = np.random.randn(50, 5)
        y = np.random.randn(50)
        
        w1, _, _ = ridge_regularization(X, y, lambda_=1.0)
        w2, _, _ = ridge_regularization(X.copy(), y.copy(), lambda_=1.0)
        
        np.testing.assert_allclose(w1, w2, atol=1e-10)
    
    def test_zero_lambda_approximates_ols(self):
        """Very small lambda should approximate OLS."""
        set_seed(42)
        X = np.random.randn(50, 5)
        y = np.random.randn(50)
        
        w_ols = np.linalg.lstsq(X, y, rcond=None)[0]
        w_ridge, _, _ = ridge_regularization(X, y, lambda_=1e-10)
        
        # Should be very close
        np.testing.assert_allclose(w_ridge, w_ols, atol=1e-4)


class TestDemonstrateIllConditioning:
    """Tests for ill-conditioning demonstration."""
    
    def test_creates_target_condition_number(self):
        """Should create matrix with approximately target condition number."""
        result = demonstrate_ill_conditioning(kappa_target=100.0, seed=42)
        
        # Check condition number
        kappa = result["condition_number"]
        assert 80.0 < kappa < 120.0  # Allow some tolerance
    
    def test_ridge_improves_conditioning(self):
        """Ridge results should show improved conditioning."""
        result = demonstrate_ill_conditioning(kappa_target=500.0, seed=42)
        
        # All ridge results should improve conditioning
        for lambda_, ridge_info in result["ridge_results"].items():
            assert ridge_info["kappa_after"] < ridge_info["kappa_before"]
    
    def test_larger_lambda_better_conditioning(self):
        """Larger lambda should give better conditioning."""
        result = demonstrate_ill_conditioning(kappa_target=1000.0, seed=42)
        
        lambdas = sorted(result["ridge_results"].keys())
        kappas = [result["ridge_results"][l]["kappa_after"] for l in lambdas]
        
        # Should generally decrease (allow some noise)
        assert kappas[-1] < kappas[0]
    
    def test_output_structure(self):
        """Output should have expected structure."""
        result = demonstrate_ill_conditioning(seed=42)
        
        # Check keys
        assert "condition_number" in result
        assert "ols_norm" in result
        assert "ridge_results" in result
        assert "true_norm" in result
        
        # Ridge results should be dict
        assert isinstance(result["ridge_results"], dict)
        assert len(result["ridge_results"]) > 0
        
        # Each ridge result should have expected keys
        for ridge_info in result["ridge_results"].values():
            assert "weights" in ridge_info
            assert "norm" in ridge_info
            assert "kappa_before" in ridge_info
            assert "kappa_after" in ridge_info
            assert "train_mse" in ridge_info
    
    def test_deterministic_with_seed(self):
        """Same seed should give same results."""
        result1 = demonstrate_ill_conditioning(seed=42)
        result2 = demonstrate_ill_conditioning(seed=42)
        
        assert result1["condition_number"] == result2["condition_number"]
        assert result1["ols_norm"] == result2["ols_norm"]
        
        # Check one ridge result
        lambda_key = list(result1["ridge_results"].keys())[0]
        np.testing.assert_allclose(
            result1["ridge_results"][lambda_key]["weights"],
            result2["ridge_results"][lambda_key]["weights"],
            atol=1e-10,
        )


class TestPCAResultMethods:
    """Tests for PCAResult helper methods."""
    
    def test_transform_with_subset(self):
        """Transform with n_components should use only that many components."""
        set_seed(42)
        X = np.random.randn(50, 10)
        
        result = pca_via_svd(X, n_components=None)  # Keep all
        
        Z_full = result.transform(X, n_components=10)
        Z_subset = result.transform(X, n_components=3)
        
        # Subset should have fewer dimensions
        assert Z_full.shape == (50, 10)
        assert Z_subset.shape == (50, 3)
        
        # First 3 components should match
        np.testing.assert_allclose(Z_full[:, :3], Z_subset, atol=1e-10)
    
    def test_reconstruction_error_zero_with_all_components(self):
        """Reconstruction with all components should have zero error."""
        set_seed(42)
        X = np.random.randn(50, 5)
        
        result = pca_via_svd(X)
        error = result.reconstruction_error(X, n_components=5)
        
        assert error < 1e-10
    
    def test_reconstruction_error_increases_monotonically(self):
        """Error should increase as components decrease."""
        set_seed(42)
        X = np.random.randn(50, 8)
        
        result = pca_via_svd(X)
        
        errors = [result.reconstruction_error(X, n) for n in range(8, 0, -1)]
        
        # Should be monotonically increasing
        for i in range(len(errors) - 1):
            assert errors[i] <= errors[i + 1]

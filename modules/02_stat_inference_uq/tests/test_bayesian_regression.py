"""Tests for Bayesian Linear Regression."""

import pytest
import numpy as np
from modules._import_helper import safe_import_from

# Python 3.12+ workaround for numeric module names
BayesianLinearRegression, posterior_predictive = safe_import_from(
    '02_stat_inference_uq.src.bayesian_regression',
    'BayesianLinearRegression', 'posterior_predictive'
)
get_rng = safe_import_from('00_repo_standards.src.mlphys_core.seeding', 'get_rng')


class TestBayesianLinearRegression:
    """Unit tests for Bayesian linear regression."""
    
    def test_fit_shapes(self):
        """Test that posterior parameters have correct shapes."""
        rng = get_rng(42)
        X = rng.standard_normal((50, 3))
        y = rng.standard_normal(50)
        
        model = BayesianLinearRegression(fit_intercept=True)
        model.fit(X, y)
        
        # With intercept, n_features = 3 + 1 = 4
        assert model.posterior_mean_.shape == (4,)
        assert model.posterior_cov_.shape == (4, 4)
        assert model.posterior_precision_.shape == (4, 4)
        
    def test_fit_no_intercept(self):
        """Test fitting without intercept."""
        rng = get_rng(42)
        X = rng.standard_normal((50, 3))
        y = rng.standard_normal(50)
        
        model = BayesianLinearRegression(fit_intercept=False)
        model.fit(X, y)
        
        assert model.posterior_mean_.shape == (3,)
        
    def test_predict_mean(self):
        """Test predictive mean computation."""
        rng = get_rng(42)
        X_train = rng.standard_normal((50, 2))
        y_train = rng.standard_normal(50)
        X_test = rng.standard_normal((10, 2))
        
        model = BayesianLinearRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        assert y_pred.shape == (10,)
        assert np.all(np.isfinite(y_pred))
        
    def test_predict_std(self):
        """Test predictive std computation."""
        rng = get_rng(42)
        X_train = rng.standard_normal((50, 2))
        y_train = rng.standard_normal(50)
        X_test = rng.standard_normal((10, 2))
        
        model = BayesianLinearRegression(noise_variance=0.5)
        model.fit(X_train, y_train)
        
        y_pred, y_std = model.predict(X_test, return_std=True)
        
        assert y_pred.shape == (10,)
        assert y_std.shape == (10,)
        assert np.all(y_std > 0)  # Std must be positive
        assert np.all(np.isfinite(y_std))
        
    def test_predict_cov(self):
        """Test predictive covariance computation."""
        rng = get_rng(42)
        X_train = rng.standard_normal((50, 2))
        y_train = rng.standard_normal(50)
        X_test = rng.standard_normal((10, 2))
        
        model = BayesianLinearRegression()
        model.fit(X_train, y_train)
        
        y_pred, y_cov = model.predict(X_test, return_cov=True)
        
        assert y_pred.shape == (10,)
        assert y_cov.shape == (10, 10)
        # Covariance must be symmetric and positive semi-definite
        assert np.allclose(y_cov, y_cov.T)
        eigvals = np.linalg.eigvalsh(y_cov)
        assert np.all(eigvals >= -1e-10)  # PSD (allow small numerical errors)
        
    def test_sample_parameters(self):
        """Test parameter sampling from posterior."""
        rng = get_rng(42)
        X_train = rng.standard_normal((50, 2))
        y_train = rng.standard_normal(50)
        
        model = BayesianLinearRegression(fit_intercept=False)
        model.fit(X_train, y_train)
        
        samples = model.sample_parameters(n_samples=100, rng=rng)
        
        assert samples.shape == (100, 2)
        assert np.all(np.isfinite(samples))
        
        # Sample mean should be close to posterior mean
        assert np.allclose(np.mean(samples, axis=0), model.posterior_mean_, atol=0.5)
        
    def test_posterior_predictive_samples(self):
        """Test posterior predictive sampling."""
        rng = get_rng(42)
        X_train = rng.standard_normal((50, 2))
        y_train = rng.standard_normal(50)
        X_test = rng.standard_normal((10, 2))
        
        model = BayesianLinearRegression(fit_intercept=False)
        model.fit(X_train, y_train)
        
        samples = posterior_predictive(model, X_test, n_samples=500, rng=rng)
        
        assert samples.shape == (500, 10)
        assert np.all(np.isfinite(samples))
        
        # Empirical mean should match predictive mean
        y_pred = model.predict(X_test)
        empirical_mean = np.mean(samples, axis=0)
        assert np.allclose(empirical_mean, y_pred, atol=0.2)
        
    def test_log_marginal_likelihood(self):
        """Test log marginal likelihood computation."""
        rng = get_rng(42)
        X = rng.standard_normal((30, 2))
        y = rng.standard_normal(30)
        
        model = BayesianLinearRegression(noise_variance=1.0)
        model.fit(X, y)
        
        log_evidence = model.log_marginal_likelihood(X, y)
        
        assert np.isfinite(log_evidence)
        assert isinstance(log_evidence, float)
        
    def test_reproducibility(self):
        """Test that predictions are reproducible."""
        rng1 = get_rng(42)
        rng2 = get_rng(42)
        
        X_train = rng1.standard_normal((50, 2))
        y_train = rng1.standard_normal(50)
        X_test = rng1.standard_normal((10, 2))
        
        # First run
        model1 = BayesianLinearRegression()
        model1.fit(X_train, y_train)
        y_pred1 = model1.predict(X_test)
        
        # Reset data generation
        rng2 = get_rng(42)
        X_train2 = rng2.standard_normal((50, 2))
        y_train2 = rng2.standard_normal(50)
        X_test2 = rng2.standard_normal((10, 2))
        
        # Second run
        model2 = BayesianLinearRegression()
        model2.fit(X_train2, y_train2)
        y_pred2 = model2.predict(X_test2)
        
        assert np.array_equal(y_pred1, y_pred2)
        
    def test_perfect_fit_low_noise(self):
        """Test that model recovers true weights with low noise."""
        rng = get_rng(42)
        n_samples = 200
        true_weights = np.array([2.0, -1.5, 0.5])
        
        X = rng.standard_normal((n_samples, 3))
        y = X @ true_weights + 0.01 * rng.standard_normal(n_samples)
        
        model = BayesianLinearRegression(noise_variance=0.01**2, fit_intercept=False)
        model.fit(X, y)
        
        # Posterior mean should be close to true weights
        assert np.allclose(model.posterior_mean_, true_weights, atol=0.1)
        
    def test_input_validation(self):
        """Test input validation."""
        model = BayesianLinearRegression()
        
        # Wrong X shape
        with pytest.raises(ValueError, match="X must be 2D"):
            model.fit(np.array([1, 2, 3]), np.array([1, 2, 3]))
            
        # Wrong y shape
        with pytest.raises(ValueError, match="y must be 1D"):
            model.fit(np.ones((10, 2)), np.ones((10, 2)))
            
        # Mismatched shapes
        with pytest.raises(ValueError, match="same n_samples"):
            model.fit(np.ones((10, 2)), np.ones(5))
            
    def test_unfitted_error(self):
        """Test error when predicting before fitting."""
        model = BayesianLinearRegression()
        
        with pytest.raises(ValueError, match="not fitted"):
            model.predict(np.ones((5, 2)))

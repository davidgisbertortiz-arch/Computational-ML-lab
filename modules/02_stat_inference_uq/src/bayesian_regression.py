"""Bayesian Linear Regression with Gaussian priors.

Implements conjugate Bayesian inference for linear regression, providing
posterior distributions over parameters and predictive distributions with uncertainty.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class BayesianLinearRegression:
    """
    Bayesian Linear Regression with Gaussian prior and likelihood.
    
    Model:
        y = X @ w + epsilon, epsilon ~ N(0, sigma^2 * I)
        Prior: w ~ N(w_prior_mean, sigma^2 * Lambda_prior^{-1})
        
    The posterior is also Gaussian (conjugate):
        w | X, y ~ N(w_post_mean, sigma^2 * Lambda_post^{-1})
        
    Args:
        prior_precision: Prior precision matrix Lambda_prior (default: identity)
        prior_mean: Prior mean w_prior_mean (default: zero)
        noise_variance: Observation noise variance sigma^2
        fit_intercept: Whether to add intercept term
        
    Attributes:
        posterior_mean_: Posterior mean of weights
        posterior_precision_: Posterior precision matrix
        posterior_cov_: Posterior covariance matrix
        noise_variance_: Noise variance (sigma^2)
        
    Example:
        >>> X_train = np.random.randn(100, 5)
        >>> y_train = X_train @ np.array([1, -0.5, 0.3, 0, 0.8]) + 0.1 * np.random.randn(100)
        >>> model = BayesianLinearRegression(noise_variance=0.1**2)
        >>> model.fit(X_train, y_train)
        >>> y_pred, y_std = model.predict(X_test, return_std=True)
    """
    
    prior_precision: Optional[np.ndarray] = None
    prior_mean: Optional[np.ndarray] = None
    noise_variance: float = 1.0
    fit_intercept: bool = True
    
    def __post_init__(self):
        """Initialize state variables."""
        self.posterior_mean_: Optional[np.ndarray] = None
        self.posterior_precision_: Optional[np.ndarray] = None
        self.posterior_cov_: Optional[np.ndarray] = None
        self.noise_variance_: float = self.noise_variance
        self.n_features_: Optional[int] = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BayesianLinearRegression":
        """
        Compute posterior distribution given data.
        
        Args:
            X: Design matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            
        Returns:
            self: Fitted model
            
        Raises:
            ValueError: If shapes don't match
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have same n_samples: {X.shape[0]} != {y.shape[0]}")
            
        # Add intercept if requested
        if self.fit_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])
            
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        
        # Set default prior if not provided
        if self.prior_precision is None:
            # Weakly informative prior: Lambda = alpha * I with small alpha
            self.prior_precision = np.eye(n_features) * 1e-4
        if self.prior_mean is None:
            self.prior_mean = np.zeros(n_features)
            
        # Posterior precision: Lambda_post = Lambda_prior + (1/sigma^2) * X^T X
        XtX = X.T @ X
        self.posterior_precision_ = self.prior_precision + XtX / self.noise_variance
        
        # Posterior covariance: Sigma_post = Lambda_post^{-1}
        self.posterior_cov_ = np.linalg.inv(self.posterior_precision_)
        
        # Posterior mean: w_post = Sigma_post (Lambda_prior w_prior + (1/sigma^2) X^T y)
        prior_term = self.prior_precision @ self.prior_mean
        data_term = X.T @ y / self.noise_variance
        self.posterior_mean_ = self.posterior_cov_ @ (prior_term + data_term)
        
        return self
        
    def predict(
        self,
        X: np.ndarray,
        return_std: bool = False,
        return_cov: bool = False,
    ) -> Tuple[np.ndarray, ...]:
        """
        Posterior predictive mean and uncertainty.
        
        For test point x, the predictive distribution is:
            p(y* | x*, X, y) = N(y* | x*^T w_post, sigma^2 + x*^T Sigma_post x*)
            
        Args:
            X: Test design matrix (n_test, n_features)
            return_std: Return predictive standard deviations
            return_cov: Return full predictive covariance matrix
            
        Returns:
            y_mean: Predictive means (n_test,)
            y_std: Predictive std devs (n_test,) if return_std=True
            y_cov: Predictive covariance (n_test, n_test) if return_cov=True
            
        Example:
            >>> y_mean, y_std = model.predict(X_test, return_std=True)
            >>> # 95% credible intervals
            >>> lower = y_mean - 1.96 * y_std
            >>> upper = y_mean + 1.96 * y_std
        """
        if self.posterior_mean_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
            
        # Add intercept if model was fitted with it
        if self.fit_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])
            
        # Predictive mean: E[y* | x*] = x*^T w_post
        y_mean = X @ self.posterior_mean_
        
        if not (return_std or return_cov):
            return y_mean
            
        # Predictive variance: Var[y* | x*] = sigma^2 + x*^T Sigma_post x*
        #   = sigma^2 (1 + x*^T (sigma^2 Lambda_post)^{-1} x*)
        if return_cov:
            # Full covariance matrix
            y_cov = self.noise_variance * (
                np.eye(X.shape[0]) + X @ self.posterior_cov_ @ X.T
            )
            return y_mean, y_cov
            
        if return_std:
            # Diagonal variances only (much faster)
            # Var[y_i] = sigma^2 + x_i^T Sigma_post x_i
            y_var = self.noise_variance * (
                1 + np.sum(X @ self.posterior_cov_ * X, axis=1)
            )
            y_std = np.sqrt(y_var)
            return y_mean, y_std
            
    def sample_parameters(self, n_samples: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Sample from posterior distribution over weights.
        
        Args:
            n_samples: Number of parameter samples
            rng: Random number generator (for reproducibility)
            
        Returns:
            samples: Parameter samples (n_samples, n_features)
            
        Example:
            >>> from modules.00_repo_standards.src.mlphys_core import get_rng
            >>> rng = get_rng(42)
            >>> w_samples = model.sample_parameters(n_samples=1000, rng=rng)
            >>> # Monte Carlo estimate of prediction uncertainty
            >>> preds = X_test @ w_samples.T  # (n_test, n_samples)
        """
        if self.posterior_mean_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        if rng is None:
            rng = np.random.default_rng()
            
        # Sample from N(w_post, Sigma_post)
        samples = rng.multivariate_normal(
            mean=self.posterior_mean_,
            cov=self.posterior_cov_,
            size=n_samples,
        )
        
        return samples
        
    def log_marginal_likelihood(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute log marginal likelihood (model evidence).
        
        This is useful for model selection/comparison.
        
        log p(y | X) = log ∫ p(y | X, w) p(w) dw
        
        Args:
            X: Design matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            
        Returns:
            log_evidence: Log marginal likelihood
        """
        if self.fit_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])
            
        n_samples = X.shape[0]
        
        # For conjugate Gaussian model, this has closed form
        # (see Bishop PRML Section 3.5)
        XtX = X.T @ X
        Lambda_post = self.prior_precision + XtX / self.noise_variance
        Sigma_post = np.linalg.inv(Lambda_post)
        
        prior_term = self.prior_precision @ self.prior_mean
        data_term = X.T @ y / self.noise_variance
        w_post = Sigma_post @ (prior_term + data_term)
        
        # Quadratic forms
        E_w = 0.5 * (
            y @ y / self.noise_variance
            + self.prior_mean @ self.prior_precision @ self.prior_mean
            - w_post @ Lambda_post @ w_post
        )
        
        # Log determinants
        sign_prior, logdet_prior = np.linalg.slogdet(self.prior_precision)
        sign_post, logdet_post = np.linalg.slogdet(Lambda_post)
        
        # Full log evidence
        log_evidence = (
            -n_samples / 2 * np.log(2 * np.pi * self.noise_variance)
            + 0.5 * (logdet_prior - logdet_post)
            - E_w
        )
        
        return log_evidence


def posterior_predictive(
    model: BayesianLinearRegression,
    X: np.ndarray,
    n_samples: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample from posterior predictive distribution.
    
    Generates samples y* ~ p(y* | x*, X, y) by:
    1. Sampling w ~ p(w | X, y)
    2. Sampling y* ~ p(y* | x*, w)
    
    Args:
        model: Fitted BayesianLinearRegression
        X: Test points (n_test, n_features)
        n_samples: Number of predictive samples
        rng: Random number generator
        
    Returns:
        samples: Predictive samples (n_samples, n_test)
        
    Example:
        >>> samples = posterior_predictive(model, X_test, n_samples=5000)
        >>> # Empirical quantiles
        >>> q_lower = np.quantile(samples, 0.025, axis=0)
        >>> q_upper = np.quantile(samples, 0.975, axis=0)
    """
    if rng is None:
        rng = np.random.default_rng()
        
    # Sample parameters from posterior
    w_samples = model.sample_parameters(n_samples=n_samples, rng=rng)
    
    # Add intercept if needed
    if model.fit_intercept:
        X_aug = np.column_stack([np.ones(X.shape[0]), X])
    else:
        X_aug = X
        
    # Compute mean predictions for each parameter sample
    # Shape: (n_samples, n_test)
    f_samples = w_samples @ X_aug.T
    
    # Add observation noise
    noise = rng.normal(0, np.sqrt(model.noise_variance), size=f_samples.shape)
    y_samples = f_samples + noise
    
    return y_samples

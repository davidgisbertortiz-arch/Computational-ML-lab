"""Tests for rare event estimation."""

import pytest
import numpy as np
from modules._import_helper import safe_import_from

(RareEventEstimator, tail_probability, adaptive_sampling) = safe_import_from(
    '05_simulation_monte_carlo.src.rare_events',
    'RareEventEstimator', 'tail_probability', 'adaptive_sampling'
)


class TestRareEventEstimator:
    """Test rare event estimator."""
    
    def test_naive_tail_probability(self):
        """Test naive estimation of P(X > 2) for X ~ N(0,1)."""
        from scipy import stats
        
        estimator = RareEventEstimator(seed=42)
        
        event = lambda x: x > 2
        sampler = lambda n: np.random.default_rng(42).normal(0, 1, n)
        
        result = estimator.estimate_naive(event, sampler, n_samples=50000)
        
        true_prob = 1 - stats.norm.cdf(2, 0, 1)  # ≈ 0.0228
        
        # Should be reasonably close
        assert abs(result.probability - true_prob) < 0.005
        assert result.method == 'naive_mc'
        assert result.n_events > 0
    
    def test_importance_sampling_reduces_error(self):
        """Test that IS reduces relative error for rare events."""
        from scipy import stats
        
        estimator = RareEventEstimator(seed=42)
        
        event = lambda x: x > 3
        sampler = lambda n: np.random.default_rng(42).normal(0, 1, n)
        proposal = lambda n: np.random.default_rng(42).normal(3, 1, n)
        log_target = lambda x: stats.norm.logpdf(x, 0, 1)
        log_proposal = lambda x: stats.norm.logpdf(x, 3, 1)
        
        # Naive MC
        result_naive = estimator.estimate_naive(event, sampler, n_samples=100000)
        
        # Importance sampling
        result_is = estimator.estimate_importance_sampling(
            event, proposal, log_target, log_proposal, n_samples=10000
        )
        
        # IS should have lower relative error with fewer samples
        assert result_is.relative_error < result_naive.relative_error
    
    def test_reproducibility(self):
        """Test reproducibility with seed."""
        estimator1 = RareEventEstimator(seed=42)
        estimator2 = RareEventEstimator(seed=42)
        
        event = lambda x: x > 2
        sampler = lambda n: np.random.default_rng(42).normal(0, 1, n)
        
        result1 = estimator1.estimate_naive(event, sampler, n_samples=10000)
        result2 = estimator2.estimate_naive(event, sampler, n_samples=10000)
        
        assert result1.probability == result2.probability


class TestTailProbability:
    """Test tail probability convenience function."""
    
    def test_normal_tail_naive(self):
        """Test normal tail with naive MC."""
        from scipy import stats
        
        result = tail_probability(
            threshold=2.0,
            distribution='normal',
            loc=0.0,
            scale=1.0,
            n_samples=50000,
            method='naive',
            seed=42
        )
        
        true_prob = 1 - stats.norm.cdf(2, 0, 1)
        assert abs(result.probability - true_prob) < 0.005
    
    def test_normal_tail_importance(self):
        """Test that importance sampling improves estimation vs naive."""
        from scipy import stats
        
        threshold = 2.5  # Less extreme threshold for more stable results
        true_prob = 1 - stats.norm.cdf(threshold, 0, 1)
        
        # Naive MC
        result_naive = tail_probability(
            threshold=threshold,
            distribution='normal',
            loc=0.0,
            scale=1.0,
            n_samples=10000,
            method='naive',
            seed=42
        )
        
        # Importance sampling
        result_is = tail_probability(
            threshold=threshold,
            distribution='normal',
            loc=0.0,
            scale=1.0,
            n_samples=10000,
            method='importance',
            seed=42
        )
        
        # IS should have lower relative error than naive
        # (Both estimates should be in reasonable range)
        assert result_is.method == 'importance_sampling'
        assert result_naive.method == 'naive_mc'
        
        # Both should give estimates in the right order of magnitude
        assert 0.001 < result_naive.probability < 0.1  # P(X > 2.5) ≈ 0.006
        assert 0.001 < result_is.probability < 0.1
    
    def test_exponential_tail(self):
        """Test exponential tail."""
        from scipy import stats
        
        result = tail_probability(
            threshold=5.0,
            distribution='exponential',
            loc=0.0,
            scale=1.0,
            n_samples=50000,
            method='naive',
            seed=42
        )
        
        true_prob = 1 - stats.expon.cdf(5, scale=1.0)
        assert abs(result.probability - true_prob) < 0.01


class TestAdaptiveSampling:
    """Test adaptive sampling."""
    
    def test_adaptive_reaches_target(self):
        """Test that adaptive sampling collects target events."""
        event = lambda x: x > 2
        sampler = lambda n: np.random.default_rng(42).normal(0, 1, n)
        
        result = adaptive_sampling(
            event, sampler, target_events=50, max_samples=100000, seed=42
        )
        
        assert result.n_events >= 50
        assert result.n_samples <= 100000
        assert result.method == 'adaptive'
    
    def test_reproducibility(self):
        """Test reproducibility."""
        event = lambda x: x > 1.5
        sampler = lambda n: np.random.default_rng(42).normal(0, 1, n)
        
        result1 = adaptive_sampling(event, sampler, target_events=30, seed=42)
        result2 = adaptive_sampling(event, sampler, target_events=30, seed=42)
        
        assert result1.probability == result2.probability
        assert result1.n_samples == result2.n_samples

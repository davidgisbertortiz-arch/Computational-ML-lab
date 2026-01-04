"""Tests for variance reduction techniques."""

import pytest
import numpy as np
from modules._import_helper import safe_import_from

(ImportanceSampler, ControlVariates, antithetic_sampling) = safe_import_from(
    '05_simulation_monte_carlo.src.variance_reduction',
    'ImportanceSampler', 'ControlVariates', 'antithetic_sampling'
)


class TestImportanceSampling:
    """Test importance sampling."""
    
    def test_standard_normal_tail(self):
        """Test P(X > 3) for X ~ N(0,1) using importance sampling."""
        from scipy import stats
        
        sampler = ImportanceSampler(seed=42)
        
        # Event: X > 3
        event = lambda x: (x > 3).astype(float)
        
        # Proposal: N(3, 1) - shifted to tail
        proposal_sample = lambda n: np.random.default_rng(42).normal(3, 1, n)
        log_target = lambda x: stats.norm.logpdf(x, 0, 1)
        log_proposal = lambda x: stats.norm.logpdf(x, 3, 1)
        
        result = sampler.estimate(
            event, proposal_sample, log_target, log_proposal, n_samples=10000
        )
        
        # True value
        true_prob = 1 - stats.norm.cdf(3, 0, 1)
        
        # Should be close
        assert abs(result.estimate - true_prob) < 0.001
    
    def test_reproducibility(self):
        """Test reproducibility with seed."""
        from scipy import stats
        
        event = lambda x: (x > 2).astype(float)
        proposal = lambda n: np.random.default_rng(42).normal(2, 1, n)
        log_target = lambda x: stats.norm.logpdf(x, 0, 1)
        log_proposal = lambda x: stats.norm.logpdf(x, 2, 1)
        
        sampler1 = ImportanceSampler(seed=42)
        result1 = sampler1.estimate(event, proposal, log_target, log_proposal, n_samples=1000)
        
        sampler2 = ImportanceSampler(seed=42)
        result2 = sampler2.estimate(event, proposal, log_target, log_proposal, n_samples=1000)
        
        assert result1.estimate == result2.estimate


class TestControlVariates:
    """Test control variates."""
    
    def test_uniform_square(self):
        """Test E[X²] for X ~ U(0,1) using control X."""
        cv = ControlVariates(seed=42)
        
        # Function: f(x) = x²
        # Control: g(x) = x with E[X] = 0.5
        result = cv.estimate(
            func=lambda x: x**2,
            control=lambda x: x,
            control_mean=0.5,
            sampler=lambda n: np.random.default_rng(42).uniform(0, 1, n),
            n_samples=10000
        )
        
        # True value: E[X²] = 1/3 for U(0,1)
        true_value = 1/3
        
        assert abs(result.estimate - true_value) < 0.01
        # Should achieve variance reduction
        assert result.variance_reduction_factor > 1.0
    
    def test_variance_reduction_factor(self):
        """Test that VRF is computed correctly."""
        cv = ControlVariates(seed=42)
        
        result = cv.estimate(
            func=lambda x: x**2,
            control=lambda x: x,
            control_mean=0.5,
            sampler=lambda n: np.random.default_rng(42).uniform(0, 1, n),
            n_samples=5000
        )
        
        # VRF should be > 1 (variance reduced)
        assert result.variance_reduction_factor is not None
        assert result.variance_reduction_factor > 1.0


class TestAntitheticSampling:
    """Test antithetic sampling."""
    
    def test_symmetric_function(self):
        """Test antithetic sampling on symmetric function."""
        # E[X²] for X ~ N(0,1) should be 1
        func = lambda x: x**2
        sampler = lambda n: np.random.default_rng(42).normal(0, 1, n)
        
        estimate, std_error = antithetic_sampling(func, sampler, n_pairs=5000, seed=42)
        
        assert abs(estimate - 1.0) < 0.05
        assert std_error > 0
    
    def test_reproducibility(self):
        """Test reproducibility."""
        func = lambda x: x**2
        sampler = lambda n: np.random.default_rng(42).normal(0, 1, n)
        
        est1, _ = antithetic_sampling(func, sampler, n_pairs=1000, seed=42)
        est2, _ = antithetic_sampling(func, sampler, n_pairs=1000, seed=42)
        
        assert est1 == est2

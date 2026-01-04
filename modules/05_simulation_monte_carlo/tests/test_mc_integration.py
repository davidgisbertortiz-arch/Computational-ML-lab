"""Tests for Monte Carlo integration."""

import pytest
import numpy as np
from modules._import_helper import safe_import_from

MCIntegrator, convergence_analysis = safe_import_from(
    '05_simulation_monte_carlo.src.mc_integration',
    'MCIntegrator', 'convergence_analysis'
)


class TestMCIntegrator:
    """Test Monte Carlo integrator."""
    
    def test_simple_integral(self):
        """Test ∫₀¹ x² dx = 1/3."""
        integrator = MCIntegrator(seed=42)
        result = integrator.integrate(
            func=lambda x: x**2,
            a=0.0,
            b=1.0,
            n_samples=10000
        )
        
        true_value = 1/3
        assert abs(result.estimate - true_value) < 0.01  # Within 1%
        assert result.std_error > 0
        assert result.ci_lower < result.estimate < result.ci_upper
    
    def test_confidence_interval_coverage(self):
        """Test that CI contains true value ~95% of time."""
        true_value = 1/3
        
        n_trials = 50
        coverage_count = 0
        
        for i in range(n_trials):
            integrator = MCIntegrator(seed=42 + i)
            result = integrator.integrate(
                func=lambda x: x**2,
                a=0.0,
                b=1.0,
                n_samples=5000
            )
            if result.ci_lower <= true_value <= result.ci_upper:
                coverage_count += 1
        
        coverage_rate = coverage_count / n_trials
        # Should be around 0.95, allow some slack
        assert 0.85 <= coverage_rate <= 1.0
    
    def test_multidim_integral(self):
        """Test 2D integral: ∫∫ x*y dx dy over [0,1]² = 1/4."""
        integrator = MCIntegrator(seed=42)
        result = integrator.integrate_multidim(
            func=lambda xy: xy[:, 0] * xy[:, 1],
            bounds=[(0, 1), (0, 1)],
            n_samples=20000
        )
        true_value = 1/4
        assert abs(result.estimate - true_value) < 0.01
    
    def test_reproducibility(self):
        """Test that same seed gives same results."""
        integrator1 = MCIntegrator(seed=42)
        integrator2 = MCIntegrator(seed=42)
        
        result1 = integrator1.integrate(
            func=lambda x: x**2,
            a=0.0,
            b=1.0,
            n_samples=1000
        )
        result2 = integrator2.integrate(
            func=lambda x: x**2,
            a=0.0,
            b=1.0,
            n_samples=1000
        )
        
        assert result1.estimate == result2.estimate
        assert result1.std_error == result2.std_error
    
    def test_convergence(self):
        """Test that error decreases with more samples."""
        n_samples_list = [100, 1000, 10000]
        errors = []
        
        true_value = 1/3
        integrator = MCIntegrator(seed=42)
        for n in n_samples_list:
            result = integrator.integrate(
                func=lambda x: x**2,
                a=0.0,
                b=1.0,
                n_samples=n
            )
            errors.append(abs(result.estimate - true_value))
        
        # Error should generally decrease
        assert errors[-1] < errors[0]
    
    def test_gaussian_integral(self):
        """Test ∫₋₅⁵ exp(-x²/2)/sqrt(2π) dx ≈ 1."""
        def gaussian(x):
            return np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
        
        integrator = MCIntegrator(seed=42)
        result = integrator.integrate(
            func=gaussian,
            a=-5.0,
            b=5.0,
            n_samples=50000
        )
        
        # Should be close to 1
        assert abs(result.estimate - 1.0) < 0.01


class TestConvergenceAnalysis:
    """Test convergence analysis."""
    
    def test_convergence_analysis(self):
        """Test convergence analysis."""
        results = convergence_analysis(
            func=lambda x: x**2,
            a=0.0,
            b=1.0,
            true_value=1/3,
            n_samples_list=[100, 500, 2000],
            n_runs=5,
            seed=42
        )
        
        assert len(results['n_samples']) == 3
        assert len(results['mean_error']) == 3
        assert len(results['std_error']) == 3
        
        # Error should decrease with more samples
        assert results['mean_error'][-1] < results['mean_error'][0]
